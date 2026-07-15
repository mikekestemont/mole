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


def binarize_image(pil_img, method: str = "sauvola", window: int = 25, k: float = 0.2):
    """Return a black-ink-on-white PIL ``L`` image for ``pil_img``."""
    import numpy as np
    from PIL import Image

    if method != "sauvola":
        raise ValueError(f"unknown binarization method {method!r} (only 'sauvola')")
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


def binarize_folder(input_dir: str | Path, out_dir: str | Path, *, method: str = "sauvola",
                    window: int = 25, k: float = 0.2, sample: int | None = None,
                    qc_html: str | Path | None = None):
    """Binarize every image in ``input_dir`` into ``out_dir`` (same filenames as PNG).

    ``sample`` limits to N random images (a quick QC preview, writes nothing to
    ``out_dir`` unless you run the full pass). Returns the per-image records.
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

    records = []
    for p in track(files, "Binarizing", unit="img"):
        orig = load_rgb(p)
        binary = binarize_image(orig, method=method, window=window, k=k)
        dst = out_dir / f"{p.stem}.png"
        if not preview:
            binary.save(dst)
        records.append({"src": p, "dst": dst, "orig": orig, "binary": binary})

    if qc_html:
        _write_qc(records, Path(qc_html), method, window, k, preview)
    # free the images we only kept for QC
    for r in records:
        r.pop("orig", None)
        r.pop("binary", None)
    return records


def _write_qc(records, out: Path, method: str, window: int, k: float, preview: bool):
    rows = []
    for r in records:
        rows.append(
            f'<tr><td class="n">{r["src"].name}</td>'
            f'<td><img src="data:image/jpeg;base64,{_thumb_b64(r["orig"])}"></td>'
            f'<td><img src="data:image/jpeg;base64,{_thumb_b64(r["binary"])}"></td></tr>')
    tag = "PREVIEW (nothing written)" if preview else "full run"
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>binarize QC</title><style>
 body{{font-family:system-ui;margin:20px;background:#111;color:#eee}}
 .meta{{color:#9ab;font-family:ui-monospace,monospace;margin-bottom:12px}}
 table{{border-collapse:collapse}} th{{color:#8bd;font-size:13px;padding:6px}}
 td{{padding:4px;text-align:center;vertical-align:top}} td.n{{font-size:10px;color:#9ab;max-width:120px;word-break:break-all}}
 td img{{width:240px;height:240px;object-fit:contain;background:#222;border-radius:4px}}</style></head><body>
<h1>Binarization QC — {method}</h1>
<div class=meta>{tag} · window={window}px · k={k} · {len(records)} images · check for broken strokes / speckle, then tune window & k</div>
<table><tr><th>file</th><th>original</th><th>binarized (black-on-white)</th></tr>{"".join(rows)}</table>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
