"""PAGE XML → text-zone regions (Transkribus / PRImA layout ground truth).

Two uses, and the first one matters more than the second:

1. **Oracle zones.** Ground-truth layout converted to a ``zones.json`` measures
   what *perfect* zone detection is worth on a collection, before anyone trains a
   detector. If the oracle buys nothing, no detector can.
2. **Detector training labels.** The same regions become YOLO boxes
   (``scripts/train_zone_detector.py``).

Namespaces are matched by LOCAL tag name, so any PAGE schema version parses
without a version table. Region polygons are quadrilaterals in these exports
(sometimes skewed), and mole's zone consumer is axis-aligned, so the polygon is
kept alongside the bbox for anything that wants it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

BBox = tuple[int, int, int, int]

# Transkribus stores a region's semantic type either as a `type` attribute or
# inside `custom="structure {type:paragraph;}"`.
_CUSTOM_TYPE = re.compile(r"structure\s*\{[^}]*type:\s*([A-Za-z_-]+)")


@dataclass
class PageRegion:
    """One region from a PAGE XML file."""

    bbox: BBox
    polygon: list[tuple[int, int]]
    tag: str                     # element local-name, e.g. "TextRegion"
    kind: str                    # semantic type if declared, else ""

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)


@dataclass
class PageLayout:
    """A parsed PAGE XML file."""

    image: str                   # imageFilename as recorded
    width: int
    height: int
    regions: list[PageRegion]
    source: Path

    def text_regions(self) -> list[PageRegion]:
        return [r for r in self.regions if r.tag == "TextRegion"]

    def text_bbox(self) -> BBox | None:
        """Axis-aligned union of the TextRegions — the main text zone."""
        rs = self.text_regions()
        if not rs:
            return None
        return (min(r.bbox[0] for r in rs), min(r.bbox[1] for r in rs),
                max(r.bbox[2] for r in rs), max(r.bbox[3] for r in rs))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _points(value: str) -> list[tuple[int, int]]:
    pts = []
    for pair in value.split():
        if "," not in pair:
            continue
        x, y = pair.split(",", 1)
        pts.append((int(round(float(x))), int(round(float(y)))))
    return pts


def read_page(xml_path: str | Path) -> PageLayout:
    """Parse one PAGE XML file."""
    xml_path = Path(xml_path)
    root = ElementTree.parse(xml_path).getroot()
    page = next((e for e in root.iter() if _local(e.tag) == "Page"), None)
    if page is None:
        raise ValueError(f"{xml_path}: no <Page> element")

    regions: list[PageRegion] = []
    for el in page.iter():
        name = _local(el.tag)
        if not name.endswith("Region"):
            continue
        coords = next((c for c in el if _local(c.tag) == "Coords"), None)
        if coords is None or not coords.get("points"):
            continue
        pts = _points(coords.get("points", ""))
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        kind = el.get("type") or ""
        if not kind:
            m = _CUSTOM_TYPE.search(el.get("custom") or "")
            kind = m.group(1) if m else ""
        regions.append(PageRegion((min(xs), min(ys), max(xs), max(ys)), pts, name, kind))

    return PageLayout(image=page.get("imageFilename") or xml_path.stem,
                      width=int(page.get("imageWidth") or 0),
                      height=int(page.get("imageHeight") or 0),
                      regions=regions, source=xml_path)


def read_page_dir(xml_dir: str | Path) -> dict[str, PageLayout]:
    """Parse a folder of PAGE XMLs, keyed by the image STEM (extension-agnostic)."""
    out: dict[str, PageLayout] = {}
    for x in sorted(Path(xml_dir).glob("*.xml")):
        try:
            layout = read_page(x)
        except (ElementTree.ParseError, ValueError):
            continue
        out[Path(layout.image).stem] = layout
    return out


def pagexml_to_zones(xml_dir: str | Path, image_dir: str | Path,
                     out: str | Path | None = None, *, padding: int = 0) -> dict:
    """Write a mole ``zones.json`` from layout ground truth — the ORACLE zones.

    Matching is by image stem, so the XML's recorded extension need not match
    what is on disk (these exports say ``.jpg`` while a binarized copy is
    ``.png``). ``padding`` grows each box, clamped to the image; zone boxes that
    clip text cost far more than boxes that keep some margin.
    """
    image_dir = Path(image_dir)
    layouts = read_page_dir(xml_dir)
    images = {p.stem: p for p in image_dir.iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}

    entries, missing = {}, []
    for stem, layout in layouts.items():
        img = images.get(stem)
        if img is None:
            missing.append(stem)
            continue
        bbox = layout.text_bbox()
        if bbox is None:
            continue
        w = layout.width or 0
        h = layout.height or 0
        x0, y0, x1, y1 = bbox
        if padding:
            x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
            x1 = min(w, x1 + padding) if w else x1 + padding
            y1 = min(h, y1 + padding) if h else y1 + padding
        entries[img.name] = {"size": [w, h],
                             "boxes": [{"bbox": [x0, y0, x1, y1],
                                        "label": "Text", "score": 1.0}]}

    manifest = {"detector": "pagexml-oracle", "source": str(xml_dir),
                "images": entries}
    if out:
        Path(out).write_text(json.dumps(manifest, indent=2))
    if missing:
        print(f"[mole] pagexml: {len(missing)} XML(s) had no matching image "
              f"(e.g. {missing[:3]})")
    print(f"[mole] pagexml: {len(entries)} zone(s) from {len(layouts)} XML file(s)")
    return manifest
