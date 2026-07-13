"""Text-zone manifest (``zones.json``): stored crop coordinates for reuse.

`mole prep` detects the main text zone of each page once and records the bounding
box (plus the raw detections) in a ``zones.json`` that lives in the dataset folder
— auto-discovered like ``labels.csv``. Downstream steps (augview, training,
embedding) read it and restrict patch-window sampling to the zone, so the YOLO
detector never runs again and no cropped images are duplicated.

The manifest is stamped with the detector + model + parameters so a zone is fully
reproducible, and it keeps per-detection info so padding / included classes can be
re-derived without re-detecting.

Format::

    {
      "meta": {"detector": "yolo", "model": "magistermilitum/YOLO_manuscripts:best.pt",
               "padding": 16, "zone_families": ["Text", "Initial"], "created": "..."},
      "images": {
        "H041r30801r.TIF": {"bbox": [481, 202, 2373, 1564], "size": [2560, 1920],
                            "fell_back": false,
                            "detections": [["Text_Main", 0.87, 500, 230, 2360, 1550], ...]}
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ZONES_FILENAME = "zones.json"

BBox = tuple[int, int, int, int]


@dataclass
class ZoneEntry:
    bbox: BBox | None                 # None => detector found no zone (fell back)
    size: tuple[int, int]             # (width, height) of the source image
    fell_back: bool = False
    detections: list[list] = field(default_factory=list)  # [label, score, x0,y0,x1,y1]


@dataclass
class ZoneManifest:
    meta: dict
    images: dict[str, ZoneEntry]

    def bbox_for(self, filename: str) -> BBox | None:
        entry = self.images.get(filename)
        return tuple(entry.bbox) if entry and entry.bbox else None


def save_zones(path: str | Path, manifest: ZoneManifest) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": manifest.meta,
        "images": {
            name: {"bbox": list(e.bbox) if e.bbox else None, "size": list(e.size),
                   "fell_back": e.fell_back, "detections": e.detections}
            for name, e in manifest.images.items()
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_zones(path: str | Path) -> ZoneManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    images = {
        name: ZoneEntry(
            bbox=tuple(v["bbox"]) if v.get("bbox") else None,
            size=tuple(v.get("size", (0, 0))),
            fell_back=v.get("fell_back", False),
            detections=v.get("detections", []),
        )
        for name, v in data.get("images", {}).items()
    }
    return ZoneManifest(meta=data.get("meta", {}), images=images)


def find_zones(dataset_root: str | Path) -> Path | None:
    """Return the dataset's ``zones.json`` path if present (auto-discovery)."""
    p = Path(dataset_root) / ZONES_FILENAME
    return p if p.is_file() else None
