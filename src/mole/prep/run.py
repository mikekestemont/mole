"""`mole prep` runner: detect the main text zone, store coordinates, QC.

Primary output is a ``zones.json`` manifest (stored coordinates for reuse) written
into the dataset folder — auto-discovered downstream like ``labels.csv``. The
detector runs ONCE; training/augview/embedding read the manifest and sample
windows inside the zone. Materialising cropped images is opt-in (``write_crops``).

Pipeline per page: detect regions -> main text zone (union of text-family boxes,
padded) -> record bbox + detections. Pages with no text zone are recorded with a
null bbox (``fell_back``) and downstream samples the whole page.
"""

from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass
from pathlib import Path

from mole.data.datasets import IMAGE_EXTENSIONS
from mole.data.zones import ZONES_FILENAME, ZoneEntry, ZoneManifest, save_zones
from mole.prep.detect import (Detection, TextZoneDetector, get_detector,
                              main_text_zone, ZONE_FAMILIES)
from mole.progress import track


@dataclass
class PrepRecord:
    """Per-page prep result, consumed by the QC sheet."""

    path: Path
    size: tuple[int, int]
    detections: list[Detection]
    zone: tuple[int, int, int, int] | None
    fell_back: bool
    crop_path: Path | None = None


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def prep_folder(input_dir: str | Path, zones_out: str | Path | None = None,
                method: str = "yolo", padding: int = 16, conf: float = 0.25,
                sample: int | None = None, zone_families: tuple[str, ...] = ZONE_FAMILIES,
                qc_html: str | Path | None = None, write_crops: str | Path | None = None,
                seed: int = 0, detector: TextZoneDetector | None = None,
                **detector_kwargs) -> tuple[ZoneManifest, list[PrepRecord]]:
    """Detect the main text zone for every page and store coordinates.

    Parameters
    ----------
    zones_out:
        Where to write ``zones.json``. Defaults to ``<input_dir>/zones.json`` so it
        is auto-discovered by augview/training.
    write_crops:
        If given, also materialise cropped images into this folder (opt-in).
    method:
        ``"yolo"`` (default) or ``"heuristic"``. Ignored if ``detector`` is given.

    Returns ``(manifest, records)``; ``records`` drive the QC sheet.
    """
    from mole.data.patches import load_rgb

    input_dir = Path(input_dir)
    files = _list_images(input_dir)
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir!r}")
    if sample is not None and sample < len(files):
        random.seed(seed)
        files = sorted(random.sample(files, sample))

    det = detector or get_detector(method, conf=conf, **detector_kwargs)
    crops_dir = Path(write_crops) if write_crops else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    records: list[PrepRecord] = []
    entries: dict[str, ZoneEntry] = {}
    for f in track(files, "Detecting text zones", unit="page"):
        img = load_rgb(f)
        dets = det.detect(f)
        zone = main_text_zone(dets, zone_families, img.size, padding=padding)
        fell_back = zone is None

        crop_path = None
        if crops_dir:
            box = zone if zone is not None else (0, 0, img.size[0], img.size[1])
            crop_path = crops_dir / f.name
            img.crop(box).save(crop_path)

        records.append(PrepRecord(path=f, size=img.size, detections=dets,
                                  zone=zone, fell_back=fell_back, crop_path=crop_path))
        entries[f.name] = ZoneEntry(
            bbox=zone, size=img.size, fell_back=fell_back,
            detections=[[d.label, round(d.score, 3), *map(int, d.bbox)] for d in dets],
        )

    manifest = ZoneManifest(
        meta={
            "detector": det.name,
            "model": getattr(det, "model_id", det.name),
            "padding": padding,
            "zone_families": list(zone_families),
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        images=entries,
    )
    zones_path = Path(zones_out) if zones_out else input_dir / ZONES_FILENAME
    save_zones(zones_path, manifest)

    from mole.prep.qc import build_contact_sheet
    if qc_html:
        build_contact_sheet(records, qc_html, detector_name=det.name)
    return manifest, records
