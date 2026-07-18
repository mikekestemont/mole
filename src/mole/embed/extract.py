"""Embedding extraction driver.

Loads a versioned checkpoint's **teacher** weights into the canonical
:class:`mole.selfsup.vit.VisionTransformer` (never a re-implemented ViT — that was
the original ``extract_embeddings.py`` divergence bug), samples zone-aware patch
windows from each page, resizes each window to ``model_size`` **deterministically**
(no train-time random-resized-crop, no raw-256 + pos-embed interpolation — the
Phase-2 resolution decision), runs the backbone, pools (mean/cls/vlad/patches),
and writes ``.npy``/parquet plus a sidecar mapping (image path -> row) stamped
with the producing model ID and embedding dim — so a FAISS index can be built on
top later.

Warns loudly if the output directory already contains embeddings from a
different model version (mixed-version indexes are the failure mode the lineage
stamping exists to prevent).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from mole.config import config_hash
from mole.data.datasets import IMAGE_EXTENSIONS
from mole.data.patches import Window, load_rgb, window_coords
from mole.data.zones import find_zones, load_zones
from mole.embed import vlad as _vlad
from mole.embed.pooling import Pooling, patch_descriptors, pool_window
from mole.progress import track

def _as_bool(v):
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


# data.* keys that --set-style overrides may change at embed time (defaults come
# from the checkpoint's training config so embed matches train unless asked).
_OVERRIDABLE = {"window_size": int, "overlap": float, "use_zones": _as_bool,
                "invert": _as_bool, "batch_size": int}


# --------------------------------------------------------------------- backbone
def load_backbone(checkpoint: str | Path, map_location: str = "cpu"):
    """Load a checkpoint's teacher ViT for inference.

    Returns ``(model, meta)`` where ``model`` is an eval-mode, grad-free
    :class:`VisionTransformer` carrying the teacher weights, and ``meta`` records
    the model ID, embedding dim, and the geometry needed to reproduce the
    training window/resize contract.
    """
    import torch

    from mole.selfsup.checkpoint import normalize_checkpoint
    from mole.selfsup.vit import build_vit

    raw = torch.load(checkpoint, map_location=map_location, weights_only=False)
    norm = normalize_checkpoint(raw)  # accepts mole or foreign (original AttMask/iBOT)
    cfg = norm["config"]
    m, d = cfg["model"], cfg["data"]
    step = norm["global_step"]
    chash = config_hash(cfg)

    from mole.selfsup.checkpoint import filtered_load

    model = build_vit(m["arch"], patch_size=m["patch_size"], return_all_tokens=True,
                      num_class_tokens=m["num_class_tokens"])
    r = filtered_load(model, norm["backbone"])
    # Only inference-irrelevant gaps are tolerated (a stray masked_embed the eval
    # model doesn't build); a real missing/mismatched weight is a hard error.
    bad = [k for k in r["missing"] + r["shape_mismatch"] if not k.startswith("masked_embed")]
    if bad:
        raise RuntimeError(f"checkpoint does not fit a {m['arch']} backbone — "
                           f"missing/mismatched: {bad[:6]}" + (" ..." if len(bad) > 6 else ""))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(map_location)

    meta = {
        "model_id": f"{m['arch']}@{chash[:8]}+step{step}",
        "config_hash": chash,
        "global_step": step,
        "arch": m["arch"],
        "patch_size": m["patch_size"],
        "num_class_tokens": m["num_class_tokens"],
        "embed_dim": int(model.embed_dim),
        "model_size": int(d["model_size"]),
        "window_size": int(d["window_size"]),
        "overlap": float(d["overlap"]),
        "use_zones": bool(d["use_zones"]),
        "invert": bool(d.get("invert", False)),
        "checkpoint": str(checkpoint),
        "source": "foreign-import" if norm["foreign"] else "mole",
    }
    return model, meta


# ------------------------------------------------------------------ page index
def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _page_index(input_dir: Path, window_size: int, overlap: float,
                use_zones: bool) -> list[tuple[Path, list[Window]]]:
    """Per-page window locations, zone-restricted, mirroring PatchWindowDataset.

    Auto-discovers ``zones.json`` (like training) and computes windows from image
    sizes alone; pixels are only touched later during extraction.
    """
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    folders = ([input_dir] + [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
               if input_dir.is_dir() else [])
    pages: list[tuple[Path, list[Window]]] = []
    for folder in folders:
        images = _list_images(folder)
        if not images:
            continue
        zpath = find_zones(folder) if use_zones else None
        manifest = load_zones(zpath) if zpath else None
        for img in images:
            bbox = manifest.bbox_for(img.name) if manifest else None
            size = manifest.images[img.name].size if (manifest and img.name in manifest.images) else None
            if not size:
                size = Image.open(img).size
            wins = window_coords(size[0], size[1], window_size, overlap, bbox)
            pages.append((img, wins))
    if not pages:
        raise FileNotFoundError(f"No images/windows found under {input_dir!r}")
    return pages


# ------------------------------------------------------------------- transform
def _build_transform(model_size: int):
    """Deterministic window -> model_size resize (BICUBIC) + ToTensor [0,1].

    Matches training's tensor contract (ToTensor, no ImageNet normalisation) but
    replaces the random-resized-crop with a plain deterministic resize.
    """
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose([
        transforms.Resize((model_size, model_size), interpolation=InterpolationMode.BICUBIC,
                           antialias=True),
        transforms.ToTensor(),
    ])


# --------------------------------------------------------------------- extract
def _page_tokens(model, crops, device, batch_size: int):
    """Run the backbone over a page's window crops -> token batch ``[W, seq, dim]``."""
    import torch

    outs = []
    for i in range(0, len(crops), batch_size):
        batch = torch.stack(crops[i:i + batch_size]).to(device)
        with torch.no_grad():
            tok = model(batch, return_attention=False, return_all_tokens=True)
        outs.append(tok.cpu())
    return torch.cat(outs, dim=0)


def _foreground_mask(crops, patch_size: int, threshold: float, method: str = "intensity"):
    """Per-patch foreground mask, aligned with :func:`patch_descriptors` order.

    CAUTION — Raven's PAPER and his RELEASED CODE specify different rules here, so
    "Raven-parity" is ambiguous and the two are offered separately. Which one produced
    the published 82.6% mAP is an open question for the author.

    ``intensity`` — Raven's ``get_foreground_mask`` **verbatim**, as released in
    ``attmask/extract_embeddings.py``::

        pooled = avg_pool2d(patches_tensor[:, 0:1], patch_size)
        foreground_mask = pooled < (1.0 - threshold)          # his default threshold=0.02

    i.e. keep patches whose mean intensity is below ``1 - threshold``. His code counts
    DARK pixels as foreground throughout (``np.sum(patch < 255)`` at window level), so
    it assumes black-ink-on-white. Consequently it is useless on parchment (background
    sits well below white, so nothing is dropped) and backwards on white-on-black
    (it would keep the background and drop the ink).

    ``raven`` — the rule as stated in the PAPER (arXiv:2409.00751), which his code does
    NOT implement: keep patch tokens whose FOREGROUND-PIXEL FRACTION is at least
    ``threshold``, with ``t_fg = 10`` foreground *pixels* per patch token (10/256 = 3.9%
    for ViT/16 — not to be confused with the paper's separate 2.5% *window* rule, see
    :func:`_window_foreground_mask`). On binarized input the per-patch mean is that
    fraction once the ink tone is known. Ink polarity is auto-detected per page as the
    minority tone (a page is never mostly ink), so unlike ``intensity`` this works on
    white-on-black and black-on-white alike. NB windows are bicubic-resized to
    ``model_size`` before this runs, so pixels are not strictly 0/1 and the mean is an
    area-averaged fraction rather than a discrete count — well preserved by the resize,
    but not a literal pixel tally.

    ``contrast`` — mole's own: keep patches whose local std exceeds ``threshold``. Text
    is high-variance strokes, blank ground is smooth. Polarity-invariant
    (``std(x) == std(1-x)``) and background-colour-agnostic, so it is the tool for
    parchment / colour photos, where pixel counting cannot work.

    EMPIRICAL NOTE (HWI, raven checkpoint): the choice barely matters. ``contrast`` and
    ``raven`` gave train mAP 0.8883 vs 0.8896 — a 0.0013 spread. Foreground selection is
    not the lever for the residual gap to the paper's numbers.
    """
    import torch

    from mole.data.patches import patch_contrast_mask

    x = torch.stack(crops)                                     # [W, C, S, S] in [0,1]
    if method == "contrast":                                   # polarity-invariant (shared helper)
        return patch_contrast_mask(x, patch_size, threshold)   # keep inked (high local std) patches
    if method not in ("intensity", "raven"):
        raise ValueError(
            f"foreground method must be 'raven', 'contrast' or 'intensity', got {method!r}")
    g = x[:, 0:1]
    mean = torch.nn.functional.avg_pool2d(g, patch_size).squeeze(1).reshape(x.shape[0], -1)
    if method == "raven":
        ink_is_bright = g.mean().item() < 0.5                  # ink = the minority tone
        frac = mean if ink_is_bright else (1.0 - mean)         # fraction of foreground pixels
        return frac >= threshold                               # PAPER: t_fg = 10 px (10/256)
    return mean < (1.0 - threshold)                            # keep non-white patches


def _mean_pool(desc: np.ndarray, with_std: bool, embed_dim: int) -> np.ndarray:
    """Codebook-free page vector from a page's foreground patch descriptors.

    ``mean``: L2-normalised mean of the tokens (``embed_dim``). ``meanstd``: the mean
    concatenated with the per-dimension std — a cheap second-order descriptor that
    keeps some of the token *distribution* VLAD encodes (``2*embed_dim``). Mean and
    std blocks are L2-normalised separately so the std contributes regardless of its
    raw magnitude. Both are fully incremental — no codebook, no cross-page fit.
    """
    if len(desc) == 0:
        return np.zeros(embed_dim * (2 if with_std else 1), dtype=np.float32)
    mu = desc.mean(0)
    mu = mu / max(float(np.linalg.norm(mu)), 1e-12)
    if not with_std:
        return mu.astype(np.float32)
    sd = desc.std(0)
    sd = sd / max(float(np.linalg.norm(sd)), 1e-12)
    return np.concatenate([mu, sd]).astype(np.float32)


def _cov_pool(desc: np.ndarray, embed_dim: int) -> np.ndarray:
    """Second-order (bilinear) page descriptor: flattened upper triangle of the token
    second-moment matrix, signed-square-rooted and L2-normalised.

    ``G = XᵀX / n`` captures cross-dimensional structure — the *shape* of the token
    cloud — that mean/meanstd (marginal only) miss, and it is codebook-free. Off-
    diagonal entries are scaled by ``√2`` so the flattened vector's inner product
    equals the full matrix's Frobenius inner product; the signed square root is the
    'improved bilinear pooling' burst-normalisation. Length is ``d(d+1)/2`` (73,920
    for d=384) — reduce with ``--whiten-dim`` for a scalable descriptor. Unimodal, so
    expected between mean and (multimodal) VLAD.
    """
    out_dim = embed_dim * (embed_dim + 1) // 2
    if len(desc) == 0:
        return np.zeros(out_dim, dtype=np.float32)
    g = (desc.T @ desc) / len(desc)                     # [d, d] second moment
    g = np.sign(g) * np.sqrt(np.abs(g))                 # signed sqrt (burst norm)
    iu = np.triu_indices(embed_dim)
    v = g[iu].astype(np.float32)
    v[iu[0] != iu[1]] *= np.float32(np.sqrt(2.0))       # off-diagonals: preserve Frobenius norm
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


def _fixed_vector_dim(pooling, embed_dim: int, num_class_tokens: int) -> int:
    """Output dim of the one-vector-per-page poolings (for the all-blank-page case)."""
    if pooling is Pooling.CLS:
        return embed_dim * num_class_tokens
    if pooling is Pooling.MEANSTD:
        return embed_dim * 2
    if pooling is Pooling.COV:
        return embed_dim * (embed_dim + 1) // 2
    return embed_dim                                    # MEAN


def _window_foreground_mask(crops, threshold: float, method: str = "raven"):
    """Raven's inference-time WINDOW pre-filter, applied to the crops *before* the ViT,
    so discarded windows cost no forward pass ("to save computation", per the paper).

    Both the paper and his released code filter windows, at slightly different values:
    the paper says keep windows with **>2.5%** foreground pixels; his
    ``attmask/extract_embeddings.py`` uses **>=2%**::

        foreground_ratio = np.sum(patch < 255) / (patch_size * patch_size)
        if foreground_ratio >= foreground_threshold:   # his default 0.02

    Note his ``patch < 255`` counts DARK pixels as foreground, i.e. black-ink-on-white.
    Here polarity is instead auto-detected once per page (the minority tone is ink), so
    either convention works; each window's foreground fraction is then its own mean.
    This threshold is separate from the per-patch rule (the paper's ``t_fg = 10`` px) —
    they are not interchangeable.

    Returns a bool tensor ``[n_windows]``. ``contrast`` uses per-window std instead,
    for non-binarized input where pixel counting is meaningless.
    """
    import torch

    x = torch.stack(crops)                                     # [W, C, S, S] in [0,1]
    g = x[:, 0:1]
    if method == "contrast":
        return g.reshape(g.shape[0], -1).std(dim=1) > threshold
    frac = g.reshape(g.shape[0], -1).mean(dim=1)
    if method == "raven":
        if g.mean().item() >= 0.5:                             # dark ink on light ground
            frac = 1.0 - frac
        return frac > threshold                                # PAPER: > 2.5% foreground
    if method != "intensity":
        raise ValueError(
            f"foreground method must be 'raven', 'contrast' or 'intensity', got {method!r}")
    return frac < (1.0 - threshold)                            # legacy: keep non-white windows


def embed(checkpoint: str | Path, input_dir: str | Path, output: str | Path,
          pooling: Pooling | str = Pooling.VLAD, whiten: bool = False,
          overrides: list[str] | None = None, *, batch_size: int = 32,
          vlad_clusters: int = 100, vlad_max_descriptors: int = 0,
          seed: int = 0, device: str | None = None,
          foreground: bool = True, foreground_threshold: float | None = None,
          foreground_method: str = "contrast",
          window_foreground: bool = False, window_foreground_threshold: float = 0.025,
          vlad_intra_norm: bool = False, invert: bool | None = None,
          codebook_from: str | Path | None = None, whiten_dim: int | None = None,
          whiten_from: str | Path | None = None):
    """Extract page embeddings for a folder of images.

    Grayscale inputs are replicated to 3 channels transparently. Output is a
    ``.npy``/parquet array plus a sidecar ``<output>.mapping.json`` recording
    image paths, embedding dim, and the producing model ID. For ``vlad`` the
    fitted codebook is saved next to the output.

    ``overrides`` accepts ``key=value`` strings for a small set of geometry knobs
    (``window_size``, ``overlap``, ``use_zones``, ``batch_size``); defaults come
    from the checkpoint so embed matches training unless deliberately changed.
    """
    import torch

    pooling = Pooling(pooling)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device) if device else _pick_device()
    model, meta = load_backbone(checkpoint, map_location=str(dev))

    if foreground_threshold is None:   # method-appropriate default (needs patch_size)
        if foreground_method == "raven":
            # PAPER: t_fg = 10 foreground PIXELS per patch token (16x16 -> 10/256 = 3.9%).
            foreground_threshold = 10.0 / float(meta["patch_size"] ** 2)
        else:
            foreground_threshold = 0.05 if foreground_method == "contrast" else 0.02

    settings = {"window_size": meta["window_size"], "overlap": meta["overlap"],
                "use_zones": meta["use_zones"], "invert": meta.get("invert", False),
                "batch_size": batch_size}
    if invert is not None:  # explicit --invert/--no-invert wins over the checkpoint's value
        settings["invert"] = invert
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        key = key.strip().removeprefix("data.")
        if key not in _OVERRIDABLE:
            raise KeyError(f"embed override {key!r} not in {sorted(_OVERRIDABLE)}")
        settings[key] = _OVERRIDABLE[key](raw)

    _warn_on_version_mismatch(output.parent, meta["model_id"])

    pages = _page_index(Path(input_dir), settings["window_size"], settings["overlap"],
                        settings["use_zones"])
    transform = _build_transform(meta["model_size"])
    nct = meta["num_class_tokens"]

    if foreground and pooling is Pooling.CLS:
        # The class token is not a patch token, so per-patch foreground selection
        # cannot apply to it (mean/meanstd/vlad/patches all honour it).
        print("[mole] note: --foreground selects patch tokens; the CLS token isn't "
              "one, so it's ignored for cls pooling")

    fg_note = (f" foreground[{foreground_method}>{foreground_threshold:g}]" if foreground else "")
    wfg_note = (f" window-fg[>{window_foreground_threshold:g}]" if window_foreground else "")
    inv_note = " inverted" if settings["invert"] else ""
    print(f"[mole] embed model={meta['model_id']} dim={meta['embed_dim']} "
          f"pooling={pooling.value}{fg_note}{wfg_note}{inv_note} device={dev} | {len(pages)} pages")
    n_win_total = n_win_kept = 0             # window pre-filter accounting

    rows: list[dict] = []
    vectors: list[np.ndarray] = []           # fixed-vector poolings (mean/cls)
    page_descriptors: list[np.ndarray] = []  # patch descriptors per page (patches/vlad)
    desc_images: list[str] = []              # image aligned with page_descriptors

    meta["invert"] = bool(settings["invert"])  # record the value actually applied

    # VLAD with an EXTERNAL codebook can encode each page on the fly and discard its
    # raw descriptors — otherwise every page's foreground tokens pile up in RAM and a
    # large test set (3600 HWI pages) OOM-kills the process. Transductive VLAD (fits a
    # codebook across all descriptors) and patches pooling still need them retained.
    stream_codebook = None
    if pooling is Pooling.VLAD and codebook_from is not None:
        stream_codebook = np.load(codebook_from).astype(np.float32)
        if stream_codebook.ndim != 2 or stream_codebook.shape[1] != meta["embed_dim"]:
            raise ValueError(
                f"codebook {codebook_from} has shape {stream_codebook.shape}; expected "
                f"(K, {meta['embed_dim']}) to match this model's {meta['embed_dim']}-dim descriptors")
        print(f"[mole] VLAD: streaming with external {stream_codebook.shape[0]}-cluster "
              f"codebook from {codebook_from} (per-page, low memory)", flush=True)
    vlad_vecs: list[np.ndarray] = []

    for img, wins in track(pages, "Embedding pages", unit="page"):
        page = load_rgb(img, invert=settings["invert"])
        crops = [transform(page.crop((w.x, w.y, w.x + w.size, w.y + w.size))) for w in wins]
        n_win_total += len(crops)
        if window_foreground and crops:      # Raven's pre-ViT window filter (saves compute)
            keep_win = _window_foreground_mask(crops, window_foreground_threshold,
                                               method=foreground_method)
            crops = [c for c, k in zip(crops, keep_win.tolist()) if k]
        n_win_kept += len(crops)
        if not crops:                        # every window was blank -> no descriptors
            if pooling in (Pooling.MEAN, Pooling.MEANSTD, Pooling.COV, Pooling.CLS):
                vectors.append(np.zeros(_fixed_vector_dim(pooling, meta["embed_dim"], nct),
                                        dtype=np.float32))
                rows.append({"row": len(rows), "image": str(img), "n_windows": 0})
            elif stream_codebook is not None:
                vlad_vecs.append(np.zeros(stream_codebook.shape[0] * meta["embed_dim"], np.float32))
                desc_images.append(str(img))
            else:
                page_descriptors.append(np.zeros((0, meta["embed_dim"]), np.float32))
                desc_images.append(str(img))
            continue
        tokens = _page_tokens(model, crops, dev, settings["batch_size"])

        if pooling is Pooling.CLS:            # class token — foreground N/A
            vec = pool_window(tokens, nct, Pooling.CLS).mean(dim=0)  # avg over windows
            vec = torch.nn.functional.normalize(vec, dim=0).numpy().astype(np.float32)
            vectors.append(vec)
            rows.append({"row": len(rows), "image": str(img), "n_windows": len(crops)})
        else:  # mean / meanstd / patches / vlad all aggregate foreground patch descriptors
            patches = patch_descriptors(tokens, nct)             # [W, num_patches, dim]
            if foreground:
                keep = _foreground_mask(crops, meta["patch_size"], foreground_threshold,
                                        method=foreground_method)
                if keep.shape[1] != patches.shape[1]:
                    raise ValueError(
                        f"foreground mask has {keep.shape[1]} patches but the model "
                        f"produced {patches.shape[1]} patch tokens per window "
                        f"(check patch_size={meta['patch_size']} / model_size={meta['model_size']})")
                desc = patches[keep].reshape(-1, meta["embed_dim"]).numpy().astype(np.float32)
            else:
                desc = patches.reshape(-1, meta["embed_dim"]).numpy().astype(np.float32)

            if pooling in (Pooling.MEAN, Pooling.MEANSTD, Pooling.COV):  # codebook-free page vector
                if pooling is Pooling.COV:
                    vectors.append(_cov_pool(desc, meta["embed_dim"]))
                else:
                    vectors.append(_mean_pool(desc, pooling is Pooling.MEANSTD, meta["embed_dim"]))
                rows.append({"row": len(rows), "image": str(img), "n_windows": len(crops)})
            elif stream_codebook is not None:    # VLAD, external codebook: encode + discard
                vlad_vecs.append(_vlad.vlad_encode(desc, stream_codebook, intra_norm=vlad_intra_norm))
                desc_images.append(str(img))
            else:                                # VLAD (transductive) / patches: retain
                page_descriptors.append(desc)
                desc_images.append(str(img))
                if pooling is Pooling.PATCHES:
                    start = len(rows)
                    rows.extend({"row": start + j, "image": str(img)} for j in range(len(desc)))

    if window_foreground and n_win_total:
        dropped = n_win_total - n_win_kept
        print(f"[mole] window-fg: kept {n_win_kept:,}/{n_win_total:,} windows "
              f"({dropped / n_win_total:.1%} skipped before the ViT)", flush=True)

    if stream_codebook is not None:
        matrix = (np.vstack(vlad_vecs) if vlad_vecs
                  else np.zeros((0, stream_codebook.shape[0] * meta["embed_dim"]), np.float32))
        codebook = stream_codebook
        rows.extend({"row": i, "image": img} for i, img in enumerate(desc_images))
    else:
        matrix, codebook = _assemble(pooling, vectors, page_descriptors, desc_images, rows,
                                     vlad_clusters, seed, intra_norm=vlad_intra_norm,
                                     codebook_from=codebook_from,
                                     max_descriptors=vlad_max_descriptors)

    whiten_transform = None
    did_whiten = (whiten or whiten_dim or whiten_from) and pooling is not Pooling.PATCHES
    if did_whiten:
        if whiten_from is not None:              # apply a transform fit on another split
            wt = np.load(whiten_from)
            whiten_apply = {"mean": wt["mean"], "proj": wt["proj"]}
            matrix = _apply_pca_whiten(matrix, whiten_apply)
            print(f"[mole] whiten: applied {whiten_apply['proj'].shape[1]}-dim transform "
                  f"from {whiten_from}", flush=True)
            meta["whiten_source"] = str(whiten_from)
        else:                                    # fit transductively (and save for reuse)
            matrix, whiten_transform = _fit_pca_whiten(matrix, dim=whiten_dim)
            meta["whiten_source"] = "fitted"
        meta["whitened"] = True
        meta["whiten_dim"] = int(matrix.shape[1])

    if window_foreground:
        meta["window_foreground"] = True
        meta["window_foreground_threshold"] = float(window_foreground_threshold)
        meta["window_kept_fraction"] = (n_win_kept / n_win_total) if n_win_total else 0.0
    _write_output(output, matrix, rows, meta, pooling, bool(did_whiten), codebook,
                  vlad_clusters, seed, foreground=foreground,
                  foreground_threshold=foreground_threshold, foreground_method=foreground_method,
                  vlad_intra_norm=vlad_intra_norm,
                  codebook_source=str(codebook_from) if codebook_from else "fitted",
                  whiten_transform=whiten_transform)
    return output


def _assemble(pooling, vectors, page_descriptors, desc_images, rows, vlad_clusters, seed,
              *, intra_norm: bool = True, codebook_from: str | Path | None = None,
              max_descriptors: int = 0):
    """Turn per-page results into the final matrix (+ codebook for vlad).

    ``rows`` is already filled for mean/cls/patches; for vlad it is empty and
    gets one row per page here (rows and the matrix stay aligned).

    VLAD fits one codebook across the embedded set's descriptors, unless
    ``codebook_from`` points at a saved ``.codebook.npy`` (e.g. one fitted on a
    training split), which is then loaded and applied — the Raven-style
    fit-on-train / apply-on-test protocol.
    """
    if pooling in (Pooling.MEAN, Pooling.MEANSTD, Pooling.COV, Pooling.CLS):
        return np.vstack(vectors), None
    if pooling is Pooling.PATCHES:
        return np.vstack(page_descriptors), None
    # VLAD: obtain a codebook (fit here, or load an external one), then encode.
    import time

    dim = page_descriptors[0].shape[1]
    if codebook_from is not None:
        codebook = np.load(codebook_from).astype(np.float32)
        if codebook.ndim != 2 or codebook.shape[1] != dim:
            raise ValueError(
                f"codebook {codebook_from} has shape {codebook.shape}; expected "
                f"(K, {dim}) to match the {dim}-dim descriptors of this model")
        print(f"[mole] VLAD: using external {codebook.shape[0]}-cluster codebook "
              f"from {codebook_from}", flush=True)
    else:
        n_desc = sum(len(d) for d in page_descriptors)
        print(f"[mole] VLAD: assembling {n_desc:,} patch descriptors from "
              f"{len(page_descriptors)} pages…", flush=True)
        all_desc = np.vstack(page_descriptors)
        print(f"[mole] VLAD: fitting {vlad_clusters}-cluster codebook on {len(all_desc):,} "
              f"patch descriptors (seed {seed})…", flush=True)
        t0 = time.perf_counter()
        codebook = _vlad.fit_codebook(all_desc, n_clusters=vlad_clusters, seed=seed,
                                      max_descriptors=max_descriptors)
        print(f"[mole] VLAD: codebook ready in {time.perf_counter() - t0:.1f}s", flush=True)
    mat = np.vstack([_vlad.vlad_encode(d, codebook, intra_norm=intra_norm)
                     for d in track(page_descriptors, "VLAD encoding", unit="page")])
    rows.extend({"row": i, "image": img} for i, img in enumerate(desc_images))
    return mat, codebook


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fit_pca_whiten(mat: np.ndarray, dim: int | None = None, eps: float = 1e-6):
    """Fit PCA-whitening on ``mat``; return ``(whitened_rows, transform)``.

    ``transform`` is ``{"mean": [D], "proj": [D, k]}`` — reusable on another matrix
    via :func:`_apply_pca_whiten` (Raven's fit-on-train / apply-on-test protocol).
    ``dim`` keeps only the top-``dim`` principal components (VLAD 38400 -> 384). The
    whitened rows are byte-for-byte what the old transductive code produced.
    """
    n = len(mat)
    mu = mat.mean(0, keepdims=True)
    x = mat - mu
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    keep = s > eps * s.max() if s.size else np.array([], dtype=bool)
    if dim is not None and keep.size:
        top = np.zeros_like(keep)
        top[:dim] = True                      # SVD returns singular values descending
        keep = keep & top
    proj = (vt[keep].T / s[keep]) * np.sqrt(max(n - 1, 1))   # [D, k]; white = x @ proj
    transform = {"mean": mu.astype(np.float32), "proj": proj.astype(np.float32)}
    return _l2(x @ proj), transform


def _apply_pca_whiten(mat: np.ndarray, transform: dict) -> np.ndarray:
    """Apply a saved PCA-whitening ``transform`` (fit elsewhere) to ``mat``, re-L2'd."""
    proj = transform["proj"]
    if proj.shape[0] != mat.shape[1]:
        raise ValueError(
            f"whiten transform expects {proj.shape[0]}-dim input but the embeddings are "
            f"{mat.shape[1]}-dim — the codebook/pooling must match the one it was fit on")
    return _l2((mat - transform["mean"]) @ proj)


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


# ----------------------------------------------------------------------- output
def _warn_on_version_mismatch(out_dir: Path, model_id: str) -> None:
    """Warn if the output dir already holds embeddings from a different model."""
    for sidecar in sorted(out_dir.glob("*.mapping.json")):
        try:
            other = json.loads(sidecar.read_text())["model_id"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        if other != model_id:
            print(f"[mole] WARNING: {sidecar.name} was produced by a DIFFERENT model "
                  f"({other} != {model_id}); mixing model versions in one index breaks "
                  f"retrieval. Use a separate output directory.")


def _write_output(output: Path, matrix, rows, meta, pooling, whiten, codebook,
                  vlad_clusters, seed, *, foreground: bool = False,
                  foreground_threshold: float = 0.02, foreground_method: str = "intensity",
                  vlad_intra_norm: bool = True, codebook_source: str = "fitted",
                  whiten_transform: dict | None = None) -> None:
    if output.suffix == ".parquet":
        _write_parquet(output, matrix, rows)
    else:
        np.save(output if output.suffix == ".npy" else output.with_suffix(".npy"), matrix)

    sidecar = {
        **meta,
        "pooling": pooling.value,
        "whiten": bool(whiten),
        "embed_matrix_shape": list(matrix.shape),
        "n_rows": len(rows),
        "foreground_filter": bool(foreground),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }
    if foreground:
        sidecar["foreground_threshold"] = float(foreground_threshold)
        sidecar["foreground_method"] = foreground_method
    if pooling is Pooling.VLAD:
        cb_path = output.with_suffix(".codebook.npy")
        np.save(cb_path, codebook)
        sidecar["vlad_clusters"] = int(codebook.shape[0])
        sidecar["vlad_seed"] = int(seed)
        sidecar["vlad_intra_norm"] = bool(vlad_intra_norm)
        sidecar["vlad_codebook_source"] = codebook_source
        sidecar["codebook"] = str(cb_path.name)
    if whiten_transform is not None:             # save so a test split can --whiten-from it
        w_path = output.with_suffix(".whiten.npz")
        np.savez(w_path, mean=whiten_transform["mean"], proj=whiten_transform["proj"])
        sidecar["whiten_transform"] = str(w_path.name)
    output.with_suffix(".mapping.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[mole] ✓ wrote {tuple(matrix.shape)} embeddings → {output}")
    print(f"[mole] ✓ sidecar → {output.with_suffix('.mapping.json')}")


def _write_parquet(output: Path, matrix, rows) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "parquet output needs pandas/pyarrow: pip install 'mole[parquet]'") from e

    df = pd.DataFrame(matrix, columns=[f"d{i}" for i in range(matrix.shape[1])])
    df.insert(0, "image", [r["image"] for r in rows])
    df.to_parquet(output)
