"""Labeled-dataset ingestion for the supervised module (Phase 1).

Builds a :class:`SupervisedIndex` over the per-dataset ``labels.csv`` defined in
:mod:`mole.data.datasets`, handling PARTIAL coverage natively (any subset of
images may be labeled; the rest are kept only as an ``unlabeled`` pool for the
later ``suggest`` path and are structurally absent from any sampler).

Three invariants make the supervised *negative rule* enforceable downstream
(see ``SUPERVISED_PLAN.md`` §0):

* **hands are namespaced** ``f"{archive}/{raw_hand}"`` so identical raw hand
  strings in two archives never collide into false positives;
* **documents are namespaced** ``f"{archive}/{doc_id}"`` (doc ids from
  :mod:`mole.data.docids`) so sibling scans of one charter share a doc and are
  never treated as cross-document positives;
* a **confidence floor** demotes low-trust rows to *unlabeled* — they then act
  as neither positives nor negatives (mirrors ``mole eval --min-confidence``).

:func:`pair_masks` is where the negative rule physically lives: positives are
same-hand / different-document, negatives are different confirmed hands, and
everything else (same document, the diagonal) is ignored. Because the sampler
draws only labeled items, an ``(labeled, unlabeled)`` pair is unrepresentable.

The window-level :class:`HandBatchSampler` is intentionally deferred to Phase 2:
it draws from the feature cache (window row-ids), which does not exist until
``build_feature_cache`` lands.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mole.data.datasets import IMAGE_EXTENSIONS, discover_datasets
from mole.data.docids import doc_id_resolver
from mole.progress import track


@dataclass
class SupItem:
    """One labeled image."""

    path: Path
    archive: str                 # dataset folder name (from discover_datasets)
    hand: str                    # NAMESPACED: f"{archive}/{raw_hand}"
    doc: str                     # NAMESPACED: f"{archive}/{doc_id}"
    confidence: float | None = None


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


@dataclass
class SupervisedIndex:
    """Labeled images grouped by (namespaced) hand and document."""

    items: list[SupItem] = field(default_factory=list)
    unlabeled: list[tuple[str, Path]] = field(default_factory=list)  # (archive, path)
    # derived (rebuilt by _reindex)
    by_hand: dict[str, list[int]] = field(default_factory=dict)
    docs_by_hand: dict[str, set[str]] = field(default_factory=dict)

    def _reindex(self) -> "SupervisedIndex":
        by_hand: dict[str, list[int]] = defaultdict(list)
        docs_by_hand: dict[str, set[str]] = defaultdict(set)
        for i, it in enumerate(self.items):
            by_hand[it.hand].append(i)
            docs_by_hand[it.hand].add(it.doc)
        self.by_hand = dict(by_hand)
        self.docs_by_hand = {h: set(d) for h, d in docs_by_hand.items()}
        return self

    @property
    def hands(self) -> list[str]:
        return sorted(self.by_hand)

    @property
    def archives(self) -> list[str]:
        return sorted({it.archive for it in self.items})

    def retrievable_hands(self, min_docs: int = 2) -> list[str]:
        """Hands that appear in at least ``min_docs`` *distinct* documents."""
        return sorted(h for h, docs in self.docs_by_hand.items()
                      if len(docs) >= min_docs)

    def subset(self, hands: set[str], *, keep_unlabeled: bool = False
               ) -> "SupervisedIndex":
        """A new index restricted to ``hands`` (indices are rebuilt)."""
        items = [it for it in self.items if it.hand in hands]
        idx = SupervisedIndex(
            items=items,
            unlabeled=list(self.unlabeled) if keep_unlabeled else [])
        return idx._reindex()

    def split_hands(self, holdout_frac: float = 0.2, seed: int = 0,
                    stratify_by_archive: bool = True
                    ) -> tuple["SupervisedIndex", "SupervisedIndex"]:
        """Partition *retrievable* hands into (train, holdout) sub-indices.

        Only hands with ≥2 documents are eligible for holdout (an unseen class
        must be queryable); every other hand — and all non-holdout retrievable
        hands — goes to train. The split is seeded and, by default, stratified
        so each archive contributes ~``holdout_frac`` of its retrievable hands.
        """
        rng = random.Random(seed)
        retrievable = self.retrievable_hands(min_docs=2)
        holdout: set[str] = set()

        groups: dict[str, list[str]]
        if stratify_by_archive:
            groups = defaultdict(list)
            for h in retrievable:
                groups[h.split("/", 1)[0]].append(h)
        else:
            groups = {"_all": list(retrievable)}

        for hs in groups.values():
            hs = sorted(hs)
            rng.shuffle(hs)
            k = round(len(hs) * holdout_frac)
            k = min(len(hs), max(1, k)) if hs else 0
            holdout.update(hs[:k])

        all_hands = set(self.by_hand)
        return self.subset(all_hands - holdout), self.subset(holdout)

    def write_holdout_split(self, path: str | Path, *, holdout_frac: float = 0.2,
                            seed: int = 0, stratify_by_archive: bool = True
                            ) -> tuple["SupervisedIndex", "SupervisedIndex"]:
        """Write a frozen split file that ``mole eval --holdout-hands`` consumes."""
        train, hold = self.split_hands(holdout_frac, seed, stratify_by_archive)
        Path(path).write_text(json.dumps({
            "seed": seed,
            "holdout_frac": holdout_frac,
            "stratify_by_archive": stratify_by_archive,
            "holdout_hands": hold.hands,
            "train_hands": train.hands,
        }, indent=2))
        return train, hold

    def stats(self) -> str:
        """Human-readable census, incl. sample doc groupings for D3 sign-off."""
        arch_imgs: Counter[str] = Counter(it.archive for it in self.items)
        arch_hands: dict[str, set[str]] = defaultdict(set)
        arch_docs: dict[str, set[str]] = defaultdict(set)
        for it in self.items:
            arch_hands[it.archive].add(it.hand)
            arch_docs[it.archive].add(it.doc)

        lines = [
            f"SupervisedIndex: {len(self.items)} labeled images, "
            f"{len(self.by_hand)} hands, {len(self.unlabeled)} unlabeled",
            "  archive: images / hands / retrievable(≥2 docs) / docs",
        ]
        for a in sorted(arch_imgs):
            retr = sum(1 for h in arch_hands[a] if len(self.docs_by_hand[h]) >= 2)
            lines.append(f"    {a}: {arch_imgs[a]} / {len(arch_hands[a])} / "
                         f"{retr} / {len(arch_docs[a])}")

        census = Counter(len(d) for d in self.docs_by_hand.values())
        lines.append("  hands by #distinct docs: "
                     + ", ".join(f"{k}→{v}" for k, v in sorted(census.items())))

        # R5 guard: show doc ids that collapse >1 image, so the grouping rules
        # can be eyeballed before any training consumes them.
        docs_to_imgs: dict[str, list[str]] = defaultdict(list)
        for it in self.items:
            docs_to_imgs[it.doc].append(it.path.name)
        multi = sorted((d, ns) for d, ns in docs_to_imgs.items() if len(ns) > 1)
        lines.append(f"  multi-image documents (siblings collapsed): {len(multi)}")
        for d, ns in multi[:10]:
            lines.append(f"    {d}: {ns}")
        if not multi:
            lines.append("    (none — every document is a single image here)")
        return "\n".join(lines)


def load_labeled_pairs(labels_root: str | Path,
                       min_confidence: float | None = None) -> SupervisedIndex:
    """Build a :class:`SupervisedIndex` from one archive folder or a pooled root.

    Walks :func:`discover_datasets` (works on the pooled symlink dir), namespaces
    hands and documents by archive, resolves doc ids via
    :func:`mole.data.docids.doc_id_resolver`, and applies the confidence floor
    (rows below it drop to the ``unlabeled`` pool — neither positive nor
    negative). Only labeled images become :class:`SupItem`\\ s.
    """
    index = SupervisedIndex()
    for m in discover_datasets(labels_root):
        archive = m.name
        table = m.labels
        resolve = doc_id_resolver(m.root)
        for img in _list_images(m.root):
            fname = img.name
            raw = table.hand_by_filename.get(fname) if table else None
            conf = table.confidence.get(fname) if table else None
            demoted = (min_confidence is not None and conf is not None
                       and conf < min_confidence)
            if raw is not None and not demoted:
                index.items.append(SupItem(
                    path=img, archive=archive,
                    hand=f"{archive}/{raw}", doc=f"{archive}/{resolve(fname)}",
                    confidence=conf))
            else:
                index.unlabeled.append((archive, img))
    return index._reindex()


def pair_masks(hands: list[str], docs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Positive / negative pair masks for a batch of labeled units.

    ``pos[i, j]`` = same hand AND different document (cross-document positives
    only). ``neg[i, j]`` = different hand (both confirmed by construction of the
    batch). Everything else — same document (incl. the diagonal) — is IGNORED:
    excluded from both masks. Unlabeled units cannot appear here at all; the
    sampler only ever draws labeled items.
    """
    h = np.asarray(hands, dtype=object)
    d = np.asarray(docs, dtype=object)
    same_hand = h[:, None] == h[None, :]
    same_doc = d[:, None] == d[None, :]
    pos = same_hand & ~same_doc          # diagonal is same_doc -> excluded
    neg = ~same_hand                     # diagonal is same_hand -> excluded
    return pos, neg


@dataclass
class FeatureCache:
    """One frozen-backbone descriptor per window, with its provenance labels.

    ``descriptors`` is ``[N_windows, dim]`` float32 (each row = mean of a
    window's foreground patch tokens). The parallel lists label each window;
    unlabeled windows carry ``hand == "" `` (kept for the ``suggest`` path, and
    structurally skipped by :class:`HandBatchSampler`). Written by
    ``build_feature_cache`` (Phase 2B) as ``cache.npy`` + ``cache.index.json``.
    """

    descriptors: np.ndarray
    window_hand: list[str]      # NAMESPACED hand, or "" if unlabeled
    window_doc: list[str]       # NAMESPACED doc,  or "" if unlabeled
    window_archive: list[str]
    window_item: list[str]      # image path / id the window came from
    meta: dict = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return len(self.descriptors)

    @property
    def dim(self) -> int:
        return int(self.descriptors.shape[1]) if len(self.descriptors) else 0

    def save(self, cache_dir: str | Path) -> Path:
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "cache.npy", self.descriptors.astype(np.float32))
        (d / "cache.index.json").write_text(json.dumps({
            "meta": self.meta,
            "item": self.window_item, "archive": self.window_archive,
            "hand": self.window_hand, "doc": self.window_doc,
        }))
        return d

    @classmethod
    def load(cls, cache_dir: str | Path) -> "FeatureCache":
        d = Path(cache_dir)
        idx = json.loads((d / "cache.index.json").read_text())
        return cls(
            descriptors=np.load(d / "cache.npy"),
            window_hand=idx["hand"], window_doc=idx["doc"],
            window_archive=idx["archive"], window_item=idx["item"],
            meta=idx.get("meta", {}))

    def filter(self, hands: set[str]) -> "FeatureCache":
        """A new cache keeping only windows whose (namespaced) hand is in ``hands``."""
        keep = [i for i, h in enumerate(self.window_hand) if h in hands]
        return FeatureCache(
            descriptors=self.descriptors[keep] if keep
            else np.zeros((0, self.dim), np.float32),
            window_hand=[self.window_hand[i] for i in keep],
            window_doc=[self.window_doc[i] for i in keep],
            window_archive=[self.window_archive[i] for i in keep],
            window_item=[self.window_item[i] for i in keep],
            meta=dict(self.meta))


class HandBatchSampler:
    """P×D×W batches over LABELED windows: the negative rule made structural.

    Each batch is ``hands_per_batch`` hands × ``docs_per_hand`` distinct
    documents × ``windows_per_doc`` windows. Only hands with ≥``docs_per_hand``
    distinct documents can be sampled (so every anchor has ≥1 cross-document
    positive; 1-doc hands are never drawn). ``same_archive_frac`` forces at least
    that fraction of a batch's hands to share one archive, so negatives are not
    dominated by the trivial cross-archive contrast (risk R1). Unlabeled windows
    are never drawn, so an ``(labeled, unlabeled)`` pair cannot enter the loss.

    Yields ``(rows, hands, docs)`` per batch: ``rows`` are cache row-ids to gather
    descriptors for; ``hands``/``docs`` are the namespaced labels aligned with
    them, ready for :func:`pair_masks`.
    """

    def __init__(self, cache: FeatureCache, *, hands_per_batch: int = 16,
                 docs_per_hand: int = 2, windows_per_doc: int = 4,
                 same_archive_frac: float = 0.5, seed: int = 0,
                 batches_per_epoch: int | None = None):
        self.cache = cache
        self.P = hands_per_batch
        self.D = docs_per_hand
        self.W = windows_per_doc
        self.same_archive_frac = same_archive_frac
        self.rng = np.random.default_rng(seed)

        by_hand_doc: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list))
        hand_archive: dict[str, str] = {}
        for i, (h, doc, arch) in enumerate(zip(cache.window_hand, cache.window_doc,
                                               cache.window_archive)):
            if not h:
                continue                                   # unlabeled: never sampled
            by_hand_doc[h][doc].append(i)
            hand_archive[h] = arch
        self.by_hand_doc = {h: dict(d) for h, d in by_hand_doc.items()}
        self.hand_archive = hand_archive
        self.anchor_hands = [h for h, docs in self.by_hand_doc.items()
                             if len(docs) >= self.D]
        if len(self.anchor_hands) < 2:
            raise ValueError(
                f"only {len(self.anchor_hands)} hand(s) have ≥{self.D} documents in "
                "the cache — cannot form contrastive batches")
        self.by_archive: dict[str, list[str]] = defaultdict(list)
        for h in self.anchor_hands:
            self.by_archive[self.hand_archive[h]].append(h)

        # a sensible default epoch length: cover every anchor hand ~once.
        self.batches_per_epoch = batches_per_epoch or max(
            1, len(self.anchor_hands) // max(1, self.P))

    def _sample_hands(self) -> list[str]:
        p = min(self.P, len(self.anchor_hands))
        n_same = int(round(p * self.same_archive_frac))
        chosen: list[str] = []
        if n_same >= 2:
            eligible = [a for a, hs in self.by_archive.items() if len(hs) >= n_same]
            if eligible:
                arch = self.rng.choice(eligible)
                chosen = list(self.rng.choice(self.by_archive[arch], size=n_same,
                                              replace=False))
        chosen = [str(h) for h in chosen]
        pool = [h for h in self.anchor_hands if h not in set(chosen)]
        need = p - len(chosen)
        if need > 0:
            chosen += [str(h) for h in self.rng.choice(
                pool, size=min(need, len(pool)), replace=False)]
        return chosen

    def _batch(self):
        rows: list[int] = []
        hands: list[str] = []
        docs: list[str] = []
        for h in self._sample_hands():
            doc_ids = list(self.by_hand_doc[h])
            picked = self.rng.choice(doc_ids, size=self.D, replace=False)
            for doc in picked:
                wins = self.by_hand_doc[h][doc]
                sel = self.rng.choice(wins, size=self.W, replace=len(wins) < self.W)
                rows.extend(int(r) for r in sel)
                hands.extend([h] * self.W)
                docs.extend([str(doc)] * self.W)
        return np.asarray(rows, dtype=np.int64), hands, docs

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            yield self._batch()


def window_descriptors(patches, keep) -> list[np.ndarray | None]:
    """One descriptor per window: mean of its foreground patch tokens.

    ``patches`` is ``[W, P, dim]`` (patch tokens per window) and ``keep`` is
    ``[W, P]`` boolean (the foreground mask). Returns a length-``W`` list; a
    window with no foreground patch yields ``None`` (skipped by the cache — it is
    a near-blank window carrying no writer signal). Accepts torch tensors or
    numpy arrays.
    """
    p = patches.detach().cpu().numpy() if hasattr(patches, "detach") else np.asarray(patches)
    k = keep.detach().cpu().numpy() if hasattr(keep, "detach") else np.asarray(keep)
    out: list[np.ndarray | None] = []
    for w in range(p.shape[0]):
        m = k[w]
        out.append(p[w][m].mean(0).astype(np.float32) if m.any() else None)
    return out


def build_feature_cache(checkpoint: str | Path, index: SupervisedIndex,
                        out_dir: str | Path, *, window_size: int = 224,
                        overlap: float = 0.0, invert: bool = True,
                        fg_method: str = "contrast", fg_threshold: float | None = None,
                        batch_size: int = 32, device: str | None = None,
                        include_unlabeled: bool = True,
                        progress: bool = True) -> FeatureCache:
    """Cache one frozen-backbone descriptor per window for every image in ``index``.

    Reuses the embed path verbatim (``load_backbone`` → deterministic window
    resize → ``_page_tokens`` → ``patch_descriptors`` → ``_foreground_mask``); the
    only new step is collapsing each window's foreground patch tokens to their
    mean (:func:`window_descriptors`). Labeled items carry their namespaced
    hand/doc; the unlabeled pool is cached too (hand/doc ``""``) for the later
    ``suggest`` path — set ``include_unlabeled=False`` to skip it, which on a
    partially-labeled pool roughly halves this pass and costs the head trainer
    nothing (:class:`HandBatchSampler` never draws unlabeled windows). Writes
    ``cache.npy`` + ``cache.index.json`` under ``out_dir``. This is the one GPU
    pass of the supervised pipeline; everything after is CPU.
    """
    import torch
    from PIL import Image, ImageFile

    from mole.data.patches import load_rgb, window_coords
    from mole.embed.extract import (
        _build_transform, _foreground_mask, _page_tokens, _pick_device, load_backbone)
    from mole.embed.pooling import patch_descriptors

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    dev = torch.device(device) if device else _pick_device()
    model, meta = load_backbone(checkpoint, map_location=str(dev))
    nct, patch_size, dim = meta["num_class_tokens"], meta["patch_size"], meta["embed_dim"]
    if fg_threshold is None:
        fg_threshold = 0.05 if fg_method == "contrast" else 0.02
    transform = _build_transform(meta["model_size"])

    entries = [(it.path, it.archive, it.hand, it.doc) for it in index.items]
    if include_unlabeled:
        entries += [(p, a, "", "") for (a, p) in index.unlabeled]

    descs: list[np.ndarray] = []
    w_hand: list[str] = []
    w_doc: list[str] = []
    w_arch: list[str] = []
    w_item: list[str] = []
    for path, archive, hand, doc in track(entries, "Caching features", unit="img",
                                          disable=not progress):
        w, h = Image.open(path).size
        wins = window_coords(w, h, window_size, overlap, None)
        if not wins:
            continue
        page = load_rgb(path, invert=invert)
        crops = [transform(page.crop((win.x, win.y, win.x + win.size, win.y + win.size)))
                 for win in wins]
        tokens = _page_tokens(model, crops, dev, batch_size)
        patches = patch_descriptors(tokens, nct)
        keep = _foreground_mask(crops, patch_size, fg_threshold, method=fg_method)
        for vec in window_descriptors(patches, keep):
            if vec is None:
                continue
            descs.append(vec)
            w_hand.append(hand); w_doc.append(doc)
            w_arch.append(archive); w_item.append(str(path))

    descriptors = (np.asarray(descs, dtype=np.float32) if descs
                   else np.zeros((0, dim), np.float32))
    cache = FeatureCache(
        descriptors, w_hand, w_doc, w_arch, w_item,
        meta={"model_id": meta["model_id"], "embed_dim": int(dim),
              "patch_size": int(patch_size), "model_size": int(meta["model_size"]),
              "window_size": int(window_size), "overlap": float(overlap),
              "invert": bool(invert), "fg_method": fg_method,
              "fg_threshold": float(fg_threshold),
              "include_unlabeled": bool(include_unlabeled),
              "base_checkpoint": str(checkpoint)})
    cache.save(out_dir)
    print(f"[mole] ✓ feature cache: {cache.n_windows:,} windows × {dim} → {out_dir}")
    return cache
