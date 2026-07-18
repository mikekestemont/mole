"""Adaptive binarization for `mole prep` (Sauvola).

Sauvola local thresholding is the historical-document standard: it adapts the
threshold per pixel from the local mean/std, so it survives the uneven
illumination and stains of camera photos where a global Otsu threshold fails.

Output is conventional **black ink on white** (so no `--invert` is needed
downstream and Raven's intensity foreground filter works). Binarized copies are
written once (cache-friendly) rather than recomputed per window at load. A QC
contact sheet shows original vs. binarized so the window/`k` params can be tuned
before committing to a whole collection.

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
                   max_side: int | None = None):
    """Return a black-ink-on-white PIL ``L`` image for ``pil_img``.

    If ``max_side`` is set, the image is downscaled (longest side, never upsampled)
    *before* thresholding, so the Sauvola window operates at the final resolution.
    """
    import numpy as np
    from PIL import Image

    if method != "sauvola":
        raise ValueError(f"unknown binarization method {method!r} (only 'sauvola')")
    pil_img = downscale_max_side(pil_img, max_side)
    gray = np.asarray(pil_img.convert("L"), dtype=np.float32)
    thresh = sauvola_threshold(gray, window=window, k=k)
    binary = np.where(gray > thresh, 255, 0).astype(np.uint8)  # bg white, ink black
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


def binarize_folder(input_dir: str | Path, out_dir: str | Path, *, method: str = "sauvola",
                    window: int = 25, k: float = 0.2, max_side: int | None = None,
                    sample: int | None = None, qc_html: str | Path | None = None):
    """Binarize every image in ``input_dir`` into ``out_dir`` (same filenames as PNG).

    ``max_side`` optionally caps the longest side (downscale-before-threshold, never
    upsample) to strip wasteful resolution. ``sample`` limits to N random images (a
    quick QC preview, writes nothing to ``out_dir`` unless you run the full pass).
    Returns the per-image records.
    """
    from mole.data.patches import load_rgb  # robust loader (multi-frame TIFF etc.)
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

    # Only keep the (heavy) orig/binary images for the rows the QC sheet will show:
    # a full 841-row contact sheet is unscrollable, slow to build, and would hold every
    # full-res original in RAM. A --sample run shows all; a full run shows an evenly-
    # spaced subset of at most QC_MAX_ROWS so it's representative of the whole corpus.
    qc_rows = set(range(len(files)))
    if qc_html and not preview and len(files) > QC_MAX_ROWS:
        step = len(files) / QC_MAX_ROWS
        qc_rows = {int(i * step) for i in range(QC_MAX_ROWS)}

    records = []
    for i, p in enumerate(track(files, "Binarizing", unit="img")):
        orig = load_rgb(p)
        binary = binarize_image(orig, method=method, window=window, k=k, max_side=max_side)
        dst = out_dir / f"{p.stem}.png"
        if not preview:
            binary.save(dst)
        rec = {"src": p, "dst": dst, "orig_size": orig.size, "final_size": binary.size}
        if qc_html and i in qc_rows:            # retain images only for QC-shown rows
            rec["orig"], rec["binary"] = orig, binary
        records.append(rec)

    if not preview:
        _carry_labels(input_dir, out_dir)
    if qc_html:
        shown = [r for r in records if "orig" in r]
        _write_qc(shown, Path(qc_html), method, window, k, max_side, preview, total=len(records))
    for r in records:                            # free the images we kept for QC
        r.pop("orig", None)
        r.pop("binary", None)
    return records


def _write_qc(records, out: Path, method: str, window: int, k: float,
              max_side: int | None, preview: bool, total: int | None = None):
    rows = []
    for r in records:
        ow, oh = r["orig_size"]
        fw, fh = r["final_size"]
        capped = " capped" if (ow, oh) != (fw, fh) else ""
        dims = f'{ow}×{oh} → {fw}×{fh}{capped}' if capped else f'{ow}×{oh}'
        rows.append(
            f'<tr><td class="n">{r["src"].name}<br><span class=d>{dims}</span></td>'
            f'<td><img src="data:image/jpeg;base64,{_thumb_b64(r["orig"])}"></td>'
            f'<td><img src="data:image/jpeg;base64,{_thumb_b64(r["binary"])}"></td>'
            f'<td><img class=detail src="data:image/png;base64,{_detail_b64(_ink_detail_crop(r["binary"]))}"></td></tr>')
    tag = "PREVIEW (nothing written)" if preview else "full run"
    cap = f"max_side={max_side}px" if max_side else "max_side=off (native resolution)"
    shown = (f"{len(records)} of {total} images (evenly-spaced sample)"
             if total and total > len(records) else f"{len(records)} images")
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>binarize QC</title><style>
 body{{font-family:system-ui;margin:20px;background:#111;color:#eee}}
 .meta{{color:#9ab;font-family:ui-monospace,monospace;margin-bottom:12px}}
 table{{border-collapse:collapse}} th{{color:#8bd;font-size:13px;padding:6px}}
 td{{padding:4px;text-align:center;vertical-align:top}} td.n{{font-size:10px;color:#9ab;max-width:120px;word-break:break-all}}
 td.n span.d{{color:#c96;font-family:ui-monospace,monospace}}
 td img{{width:240px;height:240px;object-fit:contain;background:#222;border-radius:4px}}
 td img.detail{{width:auto;height:auto;max-width:480px;max-height:480px;image-rendering:pixelated}}</style></head><body>
<h1>Binarization QC — {method}</h1>
<div class=meta>{tag} · window={window}px · k={k} · {cap} · {shown} · judge stroke crispness in the 1:1 detail column, then tune max_side / window / k</div>
<table><tr><th>file</th><th>original</th><th>binarized (black-on-white)</th><th>detail (1:1, ink-centred)</th></tr>{"".join(rows)}</table>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
