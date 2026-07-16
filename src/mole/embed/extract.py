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

    ``intensity`` (Raven et al.'s ``get_foreground_mask``): keep patches whose
    mean input intensity is below ``1 - threshold`` — drops near-white background.
    Correct for binarized / white-background scans, but USELESS on parchment,
    where the background sits well below white and nothing gets dropped.

    ``contrast``: keep patches whose local std exceeds ``threshold`` — text is
    dark strokes on a lighter ground (high variance), blank parchment is smooth
    (low variance). Background-colour-agnostic, so it removes empty patches on
    parchment / colour photos where ``intensity`` cannot.
    """
    import torch

    from mole.data.patches import patch_contrast_mask

    x = torch.stack(crops)                                     # [W, C, S, S] in [0,1]
    if method == "contrast":                                   # polarity-invariant (shared helper)
        return patch_contrast_mask(x, patch_size, threshold)   # keep inked (high local std) patches
    if method != "intensity":
        raise ValueError(f"foreground method must be 'intensity' or 'contrast', got {method!r}")
    g = x[:, 0:1]
    mean = torch.nn.functional.avg_pool2d(g, patch_size).squeeze(1).reshape(x.shape[0], -1)
    return mean < (1.0 - threshold)                            # keep non-white patches


def embed(checkpoint: str | Path, input_dir: str | Path, output: str | Path,
          pooling: Pooling | str = Pooling.VLAD, whiten: bool = False,
          overrides: list[str] | None = None, *, batch_size: int = 32,
          vlad_clusters: int = 64, seed: int = 0, device: str | None = None,
          foreground: bool = False, foreground_threshold: float | None = None,
          foreground_method: str = "intensity",
          vlad_intra_norm: bool = True, invert: bool | None = None,
          codebook_from: str | Path | None = None, whiten_dim: int | None = None):
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
    if foreground_threshold is None:   # method-appropriate default
        foreground_threshold = 0.05 if foreground_method == "contrast" else 0.02
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device) if device else _pick_device()
    model, meta = load_backbone(checkpoint, map_location=str(dev))

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

    if foreground and pooling in (Pooling.MEAN, Pooling.CLS):
        # Foreground filtering selects *patch* tokens; mean/cls pool differently
        # (mean over all patches / the class token) so it has no effect there.
        print("[mole] note: --foreground only affects patches/vlad pooling; ignored "
              f"for {pooling.value}")

    fg_note = (f" foreground[{foreground_method}>{foreground_threshold:g}]" if foreground else "")
    inv_note = " inverted" if settings["invert"] else ""
    print(f"[mole] embed model={meta['model_id']} dim={meta['embed_dim']} "
          f"pooling={pooling.value}{fg_note}{inv_note} device={dev} | {len(pages)} pages")

    rows: list[dict] = []
    vectors: list[np.ndarray] = []           # fixed-vector poolings (mean/cls)
    page_descriptors: list[np.ndarray] = []  # patch descriptors per page (patches/vlad)
    desc_images: list[str] = []              # image aligned with page_descriptors

    meta["invert"] = bool(settings["invert"])  # record the value actually applied
    for img, wins in track(pages, "Embedding pages", unit="page"):
        page = load_rgb(img, invert=settings["invert"])
        crops = [transform(page.crop((w.x, w.y, w.x + w.size, w.y + w.size))) for w in wins]
        tokens = _page_tokens(model, crops, dev, settings["batch_size"])

        if pooling in (Pooling.MEAN, Pooling.CLS):
            vec = pool_window(tokens, nct, pooling).mean(dim=0)     # avg over windows
            vec = torch.nn.functional.normalize(vec, dim=0).numpy().astype(np.float32)
            vectors.append(vec)
            rows.append({"row": len(rows), "image": str(img), "n_windows": len(wins)})
        else:  # patches / vlad both need the raw per-patch descriptors
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
            page_descriptors.append(desc)
            desc_images.append(str(img))
            if pooling is Pooling.PATCHES:
                start = len(rows)
                rows.extend({"row": start + j, "image": str(img)} for j in range(len(desc)))

    matrix, codebook = _assemble(pooling, vectors, page_descriptors, desc_images, rows,
                                 vlad_clusters, seed, intra_norm=vlad_intra_norm,
                                 codebook_from=codebook_from)

    did_whiten = (whiten or whiten_dim) and pooling is not Pooling.PATCHES
    if did_whiten:
        matrix = _pca_whiten(matrix, dim=whiten_dim)
        meta["whitened"] = True
        if whiten_dim:
            meta["whiten_dim"] = int(matrix.shape[1])

    _write_output(output, matrix, rows, meta, pooling, bool(did_whiten), codebook,
                  vlad_clusters, seed, foreground=foreground,
                  foreground_threshold=foreground_threshold, foreground_method=foreground_method,
                  vlad_intra_norm=vlad_intra_norm,
                  codebook_source=str(codebook_from) if codebook_from else "fitted")
    return output


def _assemble(pooling, vectors, page_descriptors, desc_images, rows, vlad_clusters, seed,
              *, intra_norm: bool = True, codebook_from: str | Path | None = None):
    """Turn per-page results into the final matrix (+ codebook for vlad).

    ``rows`` is already filled for mean/cls/patches; for vlad it is empty and
    gets one row per page here (rows and the matrix stay aligned).

    VLAD fits one codebook across the embedded set's descriptors, unless
    ``codebook_from`` points at a saved ``.codebook.npy`` (e.g. one fitted on a
    training split), which is then loaded and applied — the Raven-style
    fit-on-train / apply-on-test protocol.
    """
    if pooling in (Pooling.MEAN, Pooling.CLS):
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
        codebook = _vlad.fit_codebook(all_desc, n_clusters=vlad_clusters, seed=seed)
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


def _pca_whiten(mat: np.ndarray, dim: int | None = None, eps: float = 1e-6) -> np.ndarray:
    """PCA-whiten rows of ``mat`` (classic retrieval post-processing), re-L2'd.

    Fit transductively on the same matrix (standard for a one-shot index build).
    ``dim`` keeps only the top-``dim`` principal components (largest variance) —
    the reduce-and-whiten step writer retrieval uses (e.g. VLAD 38400 -> 384).
    """
    n = len(mat)
    x = mat - mat.mean(0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    keep = s > eps * s.max() if s.size else np.array([], dtype=bool)
    if dim is not None and keep.size:
        top = np.zeros_like(keep)
        top[:dim] = True                      # SVD returns singular values descending
        keep = keep & top
    white = (u * np.sqrt(max(n - 1, 1)))[:, keep]
    norms = np.linalg.norm(white, axis=1, keepdims=True)
    return (white / np.maximum(norms, 1e-12)).astype(np.float32)


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
                  vlad_intra_norm: bool = True, codebook_source: str = "fitted") -> None:
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
