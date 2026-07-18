"""QC contact sheet for `mole prep`.

One self-contained HTML page: per page, the original with every detection
overlaid (colour-coded by class, the chosen main-text zone drawn thick) next to
the resulting crop — so preprocessing quality can be eyeballed quickly. Pages
that fell back to the whole image (no text zone found) are flagged.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

# Class-family -> colour for overlay boxes.
_CLASS_COLORS = {
    "Text": "#39d353",       # main text (green)
    "Paratext": "#ff9f1c",   # marginalia / headers (orange) — excluded from zone
    "Decoration": "#4aa3ff",
    "Initial": "#4aa3ff",
    "Marks": "#b388ff",
    "Numbering": "#b388ff",
    "Damage": "#ff5c5c",
}
_ZONE_COLOR = "#ff2d55"  # chosen main-text zone (thick red)


def _family(label: str) -> str:
    return label.split("_", 1)[0]


def _color(label: str) -> str:
    return _CLASS_COLORS.get(_family(label), "#888888")


def _png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_overlay(record, box_width: int):
    """Draw detections + chosen zone on a downscaled copy of the original."""
    from PIL import Image, ImageDraw, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None       # trusted local scans can exceed PIL's ~179MP bomb limit
    img = Image.open(record.path).convert("RGB")
    w, h = img.size
    scale = box_width / w
    disp = img.resize((box_width, max(1, int(h * scale))))
    draw = ImageDraw.Draw(disp, "RGBA")

    for d in record.detections:
        col = _color(d.label)
        if d.polygon:
            pts = [(x * scale, y * scale) for x, y in d.polygon]
            draw.polygon(pts, outline=col, width=2)
        else:
            x0, y0, x1, y1 = (c * scale for c in d.bbox)
            draw.rectangle((x0, y0, x1, y1), outline=col, width=2)

    if record.zone is not None:
        x0, y0, x1, y1 = (c * scale for c in record.zone)
        draw.rectangle((x0, y0, x1, y1), outline=_ZONE_COLOR, width=4)
    return disp


def build_contact_sheet(records, output_html: str | Path, detector_name: str = "",
                        box_width: int = 460, crop_width: int = 320) -> Path:
    """Write the QC contact sheet for a list of ``PrepRecord``."""
    from PIL import Image

    from mole.progress import track

    Image.MAX_IMAGE_PIXELS = None       # trusted local scans can exceed PIL's ~179MP bomb limit
    rows = []
    for r in track(records, "Building QC sheet", unit="page"):
        overlay = _render_overlay(r, box_width)
        full = Image.open(r.path).convert("RGB")
        crop = full.crop(r.zone) if r.zone else full  # crop on the fly from bbox
        cw, ch = crop.size
        crop_disp = crop.resize((crop_width, max(1, int(ch * crop_width / cw))))

        det_summary = ", ".join(
            f'<span style="color:{_color(d.label)}">{d.label} {d.score:.2f}</span>'
            for d in sorted(r.detections, key=lambda d: -d.score)[:8]
        ) or '<span class="none">no detections</span>'
        flag = '<span class="fallback">⚠ no text zone — kept whole page</span>' if r.fell_back else ""
        zone_txt = f"zone {r.zone}" if r.zone else "—"

        rows.append(
            f'<tr><td class="meta"><b>{r.path.name}</b><br>{r.size[0]}×{r.size[1]}<br>'
            f'{zone_txt}<br>{flag}<div class="dets">{det_summary}</div></td>'
            f'<td><img src="data:image/png;base64,{_png_b64(overlay)}"></td>'
            f'<td><img class="crop" src="data:image/png;base64,{_png_b64(crop_disp)}"></td></tr>'
        )

    n_fb = sum(1 for r in records if r.fell_back)
    legend = " · ".join(f'<span style="color:{c}">■ {k}</span>' for k, c in _CLASS_COLORS.items())
    html = _QC_HTML.replace("__DETECTOR__", detector_name) \
                   .replace("__N__", str(len(records))) \
                   .replace("__NFB__", str(n_fb)) \
                   .replace("__LEGEND__", legend + f' · <span style="color:{_ZONE_COLOR}">▭ chosen zone</span>') \
                   .replace("__ROWS__", "\n".join(rows))
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


_QC_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mole prep QC</title>
<style>
 body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
 h1{font-weight:600} .sub{color:#9ab;margin:4px 0 2px}
 .legend{font-family:ui-monospace,monospace;font-size:13px;margin:8px 0 18px}
 table{border-collapse:collapse;width:100%} td{padding:8px;border-bottom:1px solid #333;vertical-align:top}
 td.meta{font-size:12px;color:#bbb;width:200px;font-family:ui-monospace,monospace}
 td img{display:block;border-radius:4px;max-width:100%;background:#222}
 img.crop{outline:2px solid #39d353}
 .fallback{color:#ff5c5c;font-weight:600}
 .dets{margin-top:8px;line-height:1.6;font-size:11px}
 .none{color:#666}
 th{color:#aaa;font-weight:500;text-align:left;padding:6px 8px}
</style></head>
<body>
 <h1>mole prep — text-zone QC</h1>
 <div class="sub">detector: <b>__DETECTOR__</b> · __N__ pages · __NFB__ fell back to whole page</div>
 <div class="legend">__LEGEND__</div>
 <table><tr><th>page</th><th>original + detections (thick red = chosen zone)</th><th>crop</th></tr>
 __ROWS__
 </table>
</body></html>
"""
