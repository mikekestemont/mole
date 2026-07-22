"""Page-level token cache — the one GPU pass that unblocks aggregator research.

:class:`FeatureCache` (``datasets.py``) stores ONE descriptor per window: the mean
of that window's foreground tokens. That is exactly the statistic mean pooling
reads out and exactly the one VLAD throws away, which is why a head trained on it
transferred under mean pooling and regressed under VLAD (SUPERVISED_PLAN.md §0a).
Anything that learns the *aggregation* therefore needs the tokens themselves.

This module caches, per page, a bounded random subsample of its foreground patch
tokens as a float16 memmap. Once it exists, fitting codebooks, training NetVLAD,
and producing page embeddings for `mole eval` are all pure CPU over the same
frozen descriptors — so an aggregator A/B is a controlled comparison (identical
tokens on both sides) instead of two GPU passes that differ in more than the
thing under test.

Why a cap is safe: VLAD is saturated in the number of tokens per page (HWI 8,800
→ 2,900 tokens/page moved mAP by +0.0006). ``max_tokens_per_page`` is a seeded
subsample, and :func:`mole.supervised.netvlad.page_vectors` can reproduce the
archive's own transductive VLAD from the cache — run that against the existing
``outputs/pooled_final/*.npy`` to *prove* the cap is harmless before trusting any
number built on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mole.progress import track

_TOKENS_FILE = "tokens.f16"
_INDEX_FILE = "tokens.index.json"


@dataclass
class TokenCache:
    """Foreground patch tokens for a set of pages, addressable per page.

    ``tokens`` is ``[N, dim]`` float16 (usually a memmap — do not assume it fits
    in RAM). ``pages`` is one record per page::

        {"item": str, "archive": str, "hand": str, "doc": str,
         "start": int, "count": int}

    ``hand``/``doc`` are namespaced ``archive/id`` exactly as in
    :class:`~mole.supervised.datasets.SupervisedIndex`; unlabeled pages carry
    ``""`` for both and are kept (the eval gallery is the *whole* archive, not
    just its labeled part).
    """

    tokens: np.ndarray
    pages: list[dict]
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ shape
    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def n_tokens(self) -> int:
        return int(len(self.tokens))

    @property
    def dim(self) -> int:
        return int(self.tokens.shape[1]) if len(self.tokens) else 0

    # ------------------------------------------------------------------- i/o
    def save(self, cache_dir: str | Path) -> Path:
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.asarray(self.tokens, dtype=np.float16).tofile(d / _TOKENS_FILE)
        (d / _INDEX_FILE).write_text(json.dumps({
            "meta": {**self.meta, "dim": self.dim, "n_tokens": self.n_tokens},
            "pages": self.pages,
        }))
        return d

    @classmethod
    def load(cls, cache_dir: str | Path, *, mmap: bool = True) -> "TokenCache":
        d = Path(cache_dir)
        idx = json.loads((d / _INDEX_FILE).read_text())
        meta = idx["meta"]
        dim, n = int(meta["dim"]), int(meta["n_tokens"])
        path = d / _TOKENS_FILE
        if mmap:
            tokens = np.memmap(path, dtype=np.float16, mode="r", shape=(n, dim))
        else:
            tokens = np.fromfile(path, dtype=np.float16).reshape(n, dim)
        return cls(tokens=tokens, pages=idx["pages"], meta=meta)

    # -------------------------------------------------------------- accessors
    def page_tokens(self, i: int, *, dtype=np.float32) -> np.ndarray:
        """The ``[count, dim]`` token block of page ``i`` (a copy, in ``dtype``)."""
        p = self.pages[i]
        return np.asarray(self.tokens[p["start"]:p["start"] + p["count"]], dtype=dtype)

    def sample_tokens(self, i: int, n: int, rng: np.random.Generator,
                      *, dtype=np.float32) -> np.ndarray:
        """``n`` tokens drawn from page ``i`` (with replacement iff it has fewer).

        A fixed ``n`` per page lets a whole batch be one ``[P, n, dim]`` tensor,
        which is what makes the differentiable aggregator a couple of matmuls.
        The count itself is not a confound: power-norm + L2 make a VLAD vector
        invariant to a global scale, and the residual sum scales linearly with
        the token count, so training on ``n`` and deploying on all of them
        differ only in sampling noise (which acts as augmentation).
        """
        p = self.pages[i]
        c = p["count"]
        if c == 0:
            return np.zeros((n, self.dim), dtype=dtype)
        sel = rng.choice(c, size=n, replace=c < n)
        block = self.tokens[p["start"]:p["start"] + c]
        return np.asarray(block[np.sort(sel)], dtype=dtype)

    def rows_for(self, *, archive: str | None = None,
                 hands: set[str] | frozenset | None = None,
                 labeled_only: bool = False) -> list[int]:
        """Page indices matching a filter (used to carve LOAO folds)."""
        out = []
        for i, p in enumerate(self.pages):
            if archive is not None and p["archive"] != archive:
                continue
            if labeled_only and not p["hand"]:
                continue
            if hands is not None and p["hand"] not in hands:
                continue
            out.append(i)
        return out

    @property
    def archives(self) -> list[str]:
        return sorted({p["archive"] for p in self.pages})

    def stats(self) -> str:
        per_archive: dict[str, list[int]] = {}
        for p in self.pages:
            per_archive.setdefault(p["archive"], []).append(p["count"])
        bits = [f"{a}: {len(c)} pages / {sum(c):,} tokens"
                for a, c in sorted(per_archive.items())]
        labeled = sum(1 for p in self.pages if p["hand"])
        return (f"{self.n_pages} pages ({labeled} labeled), {self.n_tokens:,} tokens "
                f"x {self.dim}d | " + " | ".join(bits))


def descriptor_pool(cache: TokenCache, rows: list[int] | None = None, *,
                    max_descriptors: int = 4_000_000,
                    seed: int = 0) -> np.ndarray:
    """A bounded float32 sample of tokens drawn from ``rows`` — codebook fuel.

    Sampling is proportional to page token counts (a flat draw over the pooled
    stream), matching what `mole codebook`'s reservoir does, so a codebook fit
    here is comparable to one fit by the GPU path.
    """
    rows = list(range(cache.n_pages)) if rows is None else rows
    total = sum(cache.pages[i]["count"] for i in rows)
    if total == 0:
        raise ValueError("no tokens in the requested pages")
    rng = np.random.default_rng(seed)
    keep = min(max_descriptors, total)
    frac = keep / total
    out: list[np.ndarray] = []
    for i in rows:
        block = cache.page_tokens(i)
        if len(block) == 0:
            continue
        take = int(round(len(block) * frac))
        if take <= 0:
            continue
        if take < len(block):
            block = block[rng.choice(len(block), take, replace=False)]
        out.append(block)
    return np.vstack(out) if out else np.zeros((0, cache.dim), np.float32)


def build_token_cache(checkpoint: str | Path, index, out_dir: str | Path, *,
                      window_size: int = 224, overlap: float = 0.0,
                      invert: bool = True, fg_method: str = "contrast",
                      fg_threshold: float | None = None,
                      max_tokens_per_page: int = 2048,
                      batch_size: int = 32, device: str | None = None,
                      seed: int = 0, progress: bool = True) -> TokenCache:
    """Cache foreground patch tokens for every page in ``index`` (the GPU pass).

    Reuses the embed path verbatim (``load_backbone`` → deterministic window
    resize → ``_page_tokens`` → ``patch_descriptors`` → ``_foreground_mask``), so
    the tokens are bit-identical to what `mole embed` would aggregate. Both the
    labeled items and the unlabeled pool are cached: retrieval galleries are
    whole archives, so leaving the unlabeled pages out would make the cache
    unable to reproduce an eval.

    Tokens are appended to ``tokens.f16`` as they are produced, so peak RAM is one
    page, not the corpus. Size on disk is
    ``n_pages * max_tokens_per_page * dim * 2`` bytes at worst.
    """
    import torch
    from PIL import Image, ImageFile

    from mole.data.patches import load_rgb, window_coords
    from mole.embed.extract import (
        _build_transform, _foreground_mask, _page_tokens, _pick_device, load_backbone)
    from mole.embed.pooling import patch_descriptors

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device) if device else _pick_device()
    model, meta = load_backbone(checkpoint, map_location=str(dev))
    nct, patch_size, dim = meta["num_class_tokens"], meta["patch_size"], meta["embed_dim"]
    if fg_threshold is None:
        fg_threshold = 0.05 if fg_method == "contrast" else 0.02
    transform = _build_transform(meta["model_size"])

    entries = [(it.path, it.archive, it.hand, it.doc) for it in index.items]
    entries += [(p, a, "", "") for (a, p) in index.unlabeled]

    rng = np.random.default_rng(seed)
    pages: list[dict] = []
    cursor = 0
    token_path = out_dir / _TOKENS_FILE
    with open(token_path, "wb") as fh:
        for path, archive, hand, doc in track(entries, "Caching page tokens", unit="page",
                                              disable=not progress):
            w, h = Image.open(path).size
            wins = window_coords(w, h, window_size, overlap, None)
            desc = np.zeros((0, dim), np.float32)
            if wins:
                page = load_rgb(path, invert=invert)
                crops = [transform(page.crop((win.x, win.y, win.x + win.size,
                                              win.y + win.size))) for win in wins]
                tokens = _page_tokens(model, crops, dev, batch_size)
                patches = patch_descriptors(tokens, nct)
                keep = _foreground_mask(crops, patch_size, fg_threshold, method=fg_method)
                desc = patches[keep].reshape(-1, dim).numpy().astype(np.float32)
            if max_tokens_per_page and len(desc) > max_tokens_per_page:
                sel = np.sort(rng.choice(len(desc), max_tokens_per_page, replace=False))
                desc = desc[sel]
            desc.astype(np.float16).tofile(fh)
            pages.append({"item": str(path), "archive": archive, "hand": hand,
                          "doc": doc, "start": cursor, "count": int(len(desc))})
            cursor += len(desc)

    tokens = np.memmap(token_path, dtype=np.float16, mode="r", shape=(cursor, dim))
    cache = TokenCache(tokens, pages, meta={
        "model_id": meta["model_id"], "embed_dim": int(dim),
        "patch_size": int(patch_size), "model_size": int(meta["model_size"]),
        "window_size": int(window_size), "overlap": float(overlap),
        "invert": bool(invert), "fg_method": fg_method,
        "fg_threshold": float(fg_threshold),
        "max_tokens_per_page": int(max_tokens_per_page), "seed": int(seed),
        "base_checkpoint": str(checkpoint),
    })
    (out_dir / _INDEX_FILE).write_text(json.dumps({
        "meta": {**cache.meta, "dim": int(dim), "n_tokens": int(cursor)},
        "pages": pages,
    }))
    print(f"[mole] ✓ token cache: {cache.stats()}")
    print(f"[mole]   {token_path} ({cursor * dim * 2 / 1e9:.2f} GB)")
    return cache
