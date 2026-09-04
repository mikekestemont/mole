"""Adaptive binarization for `mole prep` (Sauvola), with optional scale normalization.

Sauvola local thresholding is the historical-document standard: it adapts the
threshold per pixel from the local mean/std, so it survives the uneven
illumination and stains of camera photos where a global Otsu threshold fails.

By default a **percentile stretch** runs on the grayscale page first (interior
p2→20, p98→255; :mod:`mole.prep.stretch`). That is a per-page linear fit, so a
washed plate and a dark scan land in the same ink/paper range before the
local threshold sees them. Already-bitonal pages are skipped.

Output is conventional **black ink on white** (so no `--invert` is needed
downstream and Raven's intensity foreground filter works). Binarized copies are
written once (cache-friendly) rather than recomputed per window at load. A QC
contact sheet shows original vs. binarized so the window/`k` params can be tuned
before committing to a whole collection.

``normalize_scale`` additionally resamples each page so its **script module** is
constant across the corpus (:mod:`mole.prep.scale`) — the page is measured on
its binarization, but resampled from the *grayscale* original and re-binarized,
which is the only way to avoid aliasing bitonal strokes. The Sauvola window is
scaled along with the page so thresholding sees the same amount of *script* it
would have at the original resolution.

Implemented with scipy (already a mole dependency) — no scikit-image needed.
"""

from __future__ import annotations

import base64
import io
import os
import random
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
QC_MAX_ROWS = 40      # a full-run QC sheet shows at most this many evenly-spaced rows


def sauvola_threshold(gray, window: int = 25, k: float = 0.2, r: float = 128.0):
    """Per-pixel Sauvola threshold ``t = m * (1 + k*(s/R - 1))``.

    ``gray`` is a float array in ``[0, 255]``; ``window`` is the (odd) local
    window in px; larger ``k`` thresholds more aggressively (thinner ink).
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(gray, window, mode="reflect")
    mean_sq = uniform_filter(gray * gray, window, mode="reflect")
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))
    return mean * (1.0 + k * (std / r - 1.0))


def downscale_max_side(pil_img, max_side: int | None):
    """Downscale ``pil_img`` so its longest side is ``<= max_side`` (never upsample).

    Camera photos routinely carry far more resolution than writer retrieval needs
    (e.g. 45 MP), which only slows the CPU-bound aug pipeline. Capping here, once,
    into the cached binarized copy removes that waste for every downstream pass.
    Uses LANCZOS (high-quality) and is a no-op when the image is already smaller.
    """
    if not max_side:
        return pil_img
    from PIL import Image

    w, h = pil_img.size
    longest = max(w, h)
    if longest <= max_side:
        return pil_img
    scale = max_side / longest
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    return pil_img.resize(new, Image.LANCZOS)


def binarize_image(pil_img, method: str = "sauvola", window: int = 25, k: float = 0.2,
                   max_side: int | None = None, stretch: bool = True,
                   stretch_mask=None):
    """Return a black-ink-on-white PIL ``L`` image for ``pil_img``.

    If ``max_side`` is set, the image is downscaled (longest side, never upsampled)
    *before* thresholding, so the Sauvola window operates at the final resolution.

    ``stretch`` (default on) applies a robust percentile contrast stretch on the
    grayscale page *before* Sauvola: interior p2→20, p98→255. Faint camera plates
    get a usable ink/paper gap; already-bitonal pages are skipped. ``stretch_mask``
    is an optional boolean array (text-zone interior) used only to *estimate* the
    percentiles; the linear map is applied to the whole page.
    """
    import numpy as np
    from PIL import Image

    from mole.prep.stretch import stretch_gray

    if method != "sauvola":
        raise ValueError(f"unknown binarization method {method!r} (only 'sauvola')")
    pil_img = downscale_max_side(pil_img, max_side)
    gray = np.asarray(pil_img.convert("L"), dtype=np.uint8)
    if stretch:
        gray, _ = stretch_gray(gray, stretch_mask)
    gray_f = gray.astype(np.float32)
    thresh = sauvola_threshold(gray_f, window=window, k=k)
    binary = np.where(gray_f > thresh, 255, 0).astype(np.uint8)  # bg white, ink black
    return Image.fromarray(binary, mode="L")


def _thumb_b64(pil_img, box: int = 200) -> str:
    im = pil_img.convert("RGB").copy()
    im.thumbnail((box, box))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def _ink_detail_crop(binary, box: int = 480):
    """A native-resolution square crop centred on the ink.

    Whole-page thumbnails hide broken/merged strokes; this 1:1 window is what the
    ``--max-side`` cap and Sauvola params should actually be judged on.
    """
    import numpy as np

    arr = np.asarray(binary)  # 'L': 0 = ink (black), 255 = background (white)
    h, w = arr.shape
    ys, xs = np.nonzero(arr < 128)
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(xs) else (h // 2, w // 2)
    x0 = 0 if w <= box else max(0, min(cx - box // 2, w - box))
    y0 = 0 if h <= box else max(0, min(cy - box // 2, h - box))
    return binary.crop((x0, y0, min(w, x0 + box), min(h, y0 + box)))


def _detail_b64(pil_img) -> str:
    """PNG-encode a bitonal crop at native pixels (PNG keeps sharp edges; JPEG mushes them)."""
    buf = io.BytesIO()
    pil_img.convert("L").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _carry_labels(input_dir: Path, out_dir: Path) -> int:
    """Copy ``labels.csv`` into the binarized dataset, rewriting each image's
    extension to ``.png`` so basenames match the binarized files.

    Binarization writes ``<stem>.png`` for every image, but eval/viz/train match
    labels on the EXACT basename (extension included) — so a copied-verbatim
    ``labels.csv`` (still ``.jpg``/``.tif``) would match nothing. Only the
    extension is rewritten; every other column is preserved. Zones.json is
    deliberately NOT carried: its coordinates are in the original resolution and
    would be wrong after ``--max-side`` rescaling. Returns the row count (0 if no
    labels.csv).
    """
    import csv

    src = input_dir / "labels.csv"
    if not src.is_file():
        return 0
    with src.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = list(reader)
    if "filename" in fields:
        for r in rows:
            fn = (r.get("filename") or "").strip()
            if fn:
                r["filename"] = Path(fn).stem + ".png"
    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _odd(value: float, minimum: int = 3) -> int:
    """Nearest odd integer >= ``minimum`` (Sauvola wants an odd window)."""
    return max(minimum, int(round(value)) | 1)


def resolve_scale_target(input_dir: Path, target: float | None, *, method: str,
                         max_side: int | None, window: int, k: float,
                         sample: int) -> tuple[float, str]:
    """The module every page will be resampled to, and where it came from.

    An explicit target is used as given (that is how a second corpus is brought
    into an existing space). Otherwise it is measured from this corpus itself —
    over a deterministic sample, since a median converges long before the last
    page.
    """
    from mole.prep.scale import measure_corpus

    if target:
        return float(target), "given"
    scan = measure_corpus(input_dir, sample=sample, method=method, binarize="auto",
                          max_side=max_side, sauvola_window=window, sauvola_k=k,
                          reuse_manifest=False)
    if not scan.median:
        raise RuntimeError(
            f"could not measure a script module anywhere in {input_dir} "
            f"({scan.n_failed} pages tried) — pass an explicit target instead")
    return scan.median, f"auto:{scan.n_measured} pages"


def binarize_folder(input_dir: str | Path, out_dir: str | Path, *, method: str = "sauvola",
                    window: int = 25, k: float = 0.2, max_side: int | None = None,
                    sample: int | None = None, qc_html: str | Path | None = None,
                    normalize_scale: str = "none", target_module: float | None = None,
                    scale_sample: int | None = None, stretch: bool = True):
    """Binarize every image in ``input_dir`` into ``out_dir`` (same filenames as PNG).

    ``max_side`` optionally caps the longest side (downscale-before-threshold, never
    upsample) to strip wasteful resolution. ``sample`` limits to N random images (a
    quick QC preview, writes nothing to ``out_dir`` unless you run the full pass).

    ``stretch`` (default True) percentile-stretches grayscale before Sauvola
    (p2→20, p98→255; skipped on bitonal pages). When ``zones.json`` is present,
    percentiles are estimated inside the text-zone bbox.

    ``normalize_scale`` (``"profile"`` / ``"word"``, see :mod:`mole.prep.scale`)
    additionally resamples each page to a constant script module: ``target_module``
    px, or the corpus's own median if not given. Every page's module before and
    after is recorded in ``<out_dir>/scale.json``, which is both the provenance and
    the evidence that the scales collapsed.

    Returns the per-image records.
    """
    from mole.data.patches import load_rgb  # robust loader (multi-frame TIFF etc.)
    from mole.data.zones import find_zones, load_zones
    from mole.prep import scale as _scale
    from mole.prep.stretch import bbox_mask
    from mole.progress import track

    input_dir, out_dir = Path(input_dir), Path(out_dir)
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"no images in {input_dir}")
    preview = sample is not None
    if preview:
        random.seed(0)
        files = random.sample(files, min(sample, len(files)))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    normalizing = normalize_scale != "none"
    target, target_source = None, ""
    if normalizing:
        if normalize_scale not in _scale.METHODS:
            raise ValueError(f"normalize_scale must be 'none' or one of {_scale.METHODS}, "
                             f"got {normalize_scale!r}")
        target, target_source = resolve_scale_target(
            input_dir, target_module, method=normalize_scale, max_side=max_side,
            window=window, k=k, sample=scale_sample or _scale.DEFAULT_SAMPLE)
        print(f"[mole] scale: normalizing to a {target:.1f}px script module ({target_source})")
    zpath = find_zones(input_dir)
    zones = load_zones(zpath) if zpath else None

    # Only keep the (heavy) orig/binary images for the rows the QC sheet will show:
    # a full 841-row contact sheet is unscrollable, slow to build, and would hold every
    # full-res original in RAM. A --sample run shows all; a full run shows an evenly-
    # spaced subset of at most QC_MAX_ROWS so it's representative of the whole corpus.
    qc_rows = set(range(len(files)))
    if qc_html and not preview and len(files) > QC_MAX_ROWS:
        step = len(files) / QC_MAX_ROWS
        qc_rows = {int(i * step) for i in range(QC_MAX_ROWS)}

    records = []
    manifest = _scale.ScaleManifest()
    for i, p in enumerate(track(files, "Binarizing", unit="img")):
        orig = load_rgb(p)
        capped = downscale_max_side(orig, max_side)
        zone = _scale.zone_for(zones, p.name, capped.size)
        mask = bbox_mask(capped.size[::-1], zone) if stretch else None
        binary = binarize_image(capped, method=method, window=window, k=k,
                                stretch=stretch, stretch_mask=mask)
        rec = {"src": p, "dst": out_dir / f"{p.stem}.png", "orig_size": orig.size,
               "final_size": binary.size, "stretched": bool(stretch)}
        if normalizing:
            est = _scale.estimate_module(binary, zone, method=normalize_scale)
            factor = _scale.scale_factor(est.module, target)
            if est.module is not None and abs(factor - 1.0) >= _scale.RESIZE_EPS:
                # Resample the GRAYSCALE original and threshold again: resampling a
                # bitonal image aliases its strokes. The Sauvola window rides along
                # with the page so it still spans the same amount of script.
                resampled = _scale.resample(capped, factor)
                rzone = _scale.scaled_zone(zone, factor)
                rmask = bbox_mask(resampled.size[::-1], rzone) if stretch else None
                binary = binarize_image(resampled, method=method,
                                        window=_odd(window * factor), k=k,
                                        stretch=stretch, stretch_mask=rmask)
                out_est = _scale.estimate_module(binary, rzone, method=normalize_scale)
            else:
                factor, out_est = 1.0, est
            rec.update(module=est.module, scale=factor, module_out=out_est.module,
                       final_size=binary.size)
            manifest.images[rec["dst"].name] = _scale.ScaleEntry(
                module=est.module, scale=factor, size=binary.size, pitch=est.pitch,
                confidence=round(est.confidence, 3), module_out=out_est.module)
        if not preview:
            binary.save(rec["dst"])
        if qc_html and i in qc_rows:            # retain images only for QC-shown rows
            rec["orig"], rec["binary"] = orig, binary
        records.append(rec)

    if not preview:
        _carry_labels(input_dir, out_dir)
        if normalizing:
            manifest.meta = _scale.scale_meta(
                target, normalize_scale, target_source,
                **_collapse_stats(records), max_side=max_side or None)
            _scale.save_scale(out_dir / _scale.SCALE_FILENAME, manifest)
    if qc_html:
        shown = [r for r in records if "orig" in r]
        _write_qc(shown, Path(qc_html), method, window, k, max_side, preview,
                  total=len(records), scale_target=target,
                  scale_stats=_collapse_stats(records) if normalizing else None,
                  stretch=stretch)
    for r in records:                            # free the images we kept for QC
        r.pop("orig", None)
        r.pop("binary", None)
    return records


def _collapse_stats(records: list[dict]) -> dict:
    """Corpus module before vs after — the scale-collapse evidence, in one dict."""
    import numpy as np

    before = [r["module"] for r in records if r.get("module")]
    after = [r["module_out"] for r in records if r.get("module_out")]
    from mole.prep.scale import RESIZE_EPS

    stats: dict = {"pages": len(records), "measured": len(before),
                   "unmeasurable": len(records) - len(before),
                   "rescaled": sum(1 for r in records
                                   if abs(r.get("scale", 1.0) - 1.0) >= RESIZE_EPS)}
    for name, values in (("before", before), ("after", after)):
        if values:
            v = np.asarray(values, dtype=float)
            med = float(np.median(v))
            stats[f"median_{name}"] = round(med, 2)
            stats[f"spread_{name}"] = round(float(np.subtract(*np.percentile(v, [75, 25]))) / med, 3)
    return stats


def _scale_cell(rec: dict) -> str:
    """The per-page scale story: module measured, factor applied, module achieved."""
    if "scale" not in rec:
        return ""
    if rec.get("module") is None:
        return '<br><span class=s title="left at its original scale">module ?</span>'
    out = f' → {rec["module_out"]:.0f}' if rec.get("module_out") else ""
    return (f'<br><span class=s>module {rec["module"]:.0f}{out}px '
            f'×{rec["scale"]:.2f}</span>')


def _write_qc(records, out: Path, method: str, window: int, k: float,
              max_side: int | None, preview: bool, total: int | None = None,
              scale_target: float | None = None, scale_stats: dict | None = None,
              stretch: bool = True):
    rows = []
    for r in records:
        ow, oh = r["orig_size"]
        fw, fh = r["final_size"]
        capped = " capped" if (ow, oh) != (fw, fh) else ""
        dims = f'{ow}×{oh} → {fw}×{fh}{capped}' if capped else f'{ow}×{oh}'
        rows.append(
            f'<tr><td class="n">{r["src"].name}<br><span class=d>{dims}</span>{_scale_cell(r)}</td>'
            f'<td><img src="data:image/jpeg;base64,{_thumb_b64(r["orig"])}"></td>'
            f'<td><img src="data:image/jpeg;base64,{_thumb_b64(r["binary"])}"></td>'
            f'<td><img class=detail src="data:image/png;base64,{_detail_b64(_ink_detail_crop(r["binary"]))}"></td></tr>')
    tag = "PREVIEW (nothing written)" if preview else "full run"
    cap = f"max_side={max_side}px" if max_side else "max_side=off (native resolution)"
    cap += " · stretch p2→20/p98→255" if stretch else " · stretch off"
    shown = (f"{len(records)} of {total} images (evenly-spaced sample)"
             if total and total > len(records) else f"{len(records)} images")
    if scale_target:
        s = scale_stats or {}
        pages, rescaled = s.get("pages", 0), s.get("rescaled", 0)
        # every page lands in exactly one of these three buckets, so they add up
        fate = [f"{rescaled}/{pages} resampled"]
        if (on_target := s.get("measured", 0) - rescaled) > 0:
            fate.append(f"{on_target} already on target")
        if s.get("unmeasurable"):
            fate.append(f"{s['unmeasurable']} unmeasurable, left as-is")
        collapse = (f" · module {s.get('median_before', '?')}→{s.get('median_after', '?')}px, "
                    f"spread {s.get('spread_before', '?')}→{s.get('spread_after', '?')} "
                    f"(IQR/median) · {' · '.join(fate)}")
        cap += f" · scale-normalized to {scale_target:.1f}px{collapse}"
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>binarize QC</title><style>
 body{{font-family:system-ui;margin:20px;background:#111;color:#eee}}
 .meta{{color:#9ab;font-family:ui-monospace,monospace;margin-bottom:12px}}
 table{{border-collapse:collapse}} th{{color:#8bd;font-size:13px;padding:6px}}
 td{{padding:4px;text-align:center;vertical-align:top}} td.n{{font-size:10px;color:#9ab;max-width:120px;word-break:break-all}}
 td.n span.d{{color:#c96;font-family:ui-monospace,monospace}}
 td.n span.s{{color:#6c9;font-family:ui-monospace,monospace}}
 td img{{width:240px;height:240px;object-fit:contain;background:#222;border-radius:4px}}
 td img.detail{{width:auto;height:auto;max-width:480px;max-height:480px;image-rendering:pixelated}}</style></head><body>
<h1>Binarization QC — {method}</h1>
<div class=meta>{tag} · window={window}px · k={k} · {cap} · {shown} · judge stroke crispness in the 1:1 detail column, then tune max_side / window / k</div>
<table><tr><th>file</th><th>original</th><th>binarized (black-on-white)</th><th>detail (1:1, ink-centred)</th></tr>{"".join(rows)}</table>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
