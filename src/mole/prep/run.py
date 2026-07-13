"""`mole prep` runner: detect the main text zone, crop it, write a QC sheet.

Pipeline per page: detect regions -> take the main text zone (union of text-class
boxes, padded) -> crop -> save. Falls back to the whole page when no text zone is
found (and flags it in the QC sheet). The cropped folder is a normal dataset that
train/embed consume directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from mole.data.datasets import IMAGE_EXTENSIONS
from mole.prep.detect import (Detection, TextZoneDetector, get_detector,
                              main_text_zone, ZONE_FAMILIES)


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


def prep_folder(input_dir: str | Path, output_dir: str | Path,
                method: str = "yolo", padding: int = 16, sample: int | None = None,
                zone_families: tuple[str, ...] = ZONE_FAMILIES,
                qc_html: str | Path | None = None, seed: int = 0,
                detector: TextZoneDetector | None = None, **detector_kwargs) -> list[PrepRecord]:
    """Detect + crop the main text zone for every page in ``input_dir``.

    Parameters
    ----------
    method:
        ``"yolo"`` (default) or ``"heuristic"``. Ignored if ``detector`` is given.
    padding:
        Pixels added around the detected text zone before cropping.
    sample:
        If set, process only a random ``sample`` pages (for a quick QC pass).
    qc_html:
        Where to write the QC contact sheet. Defaults to ``<output_dir>/qc.html``.

    Returns the list of :class:`PrepRecord` (also drives the QC sheet).
    """
    from mole.data.patches import load_rgb

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    files = _list_images(input_dir)
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir!r}")
    if sample is not None and sample < len(files):
        random.seed(seed)
        files = sorted(random.sample(files, sample))

    det = detector or get_detector(method, **detector_kwargs)

    records: list[PrepRecord] = []
    for f in files:
        img = load_rgb(f)
        dets = det.detect(f)
        zone = main_text_zone(dets, zone_families, img.size, padding=padding)
        fell_back = zone is None
        crop_box = zone if zone is not None else (0, 0, img.size[0], img.size[1])
        crop = img.crop(crop_box)
        crop_path = images_out / f.name
        crop.save(crop_path)
        records.append(PrepRecord(path=f, size=img.size, detections=dets,
                                  zone=zone, fell_back=fell_back, crop_path=crop_path))

    from mole.prep.qc import build_contact_sheet
    qc_path = Path(qc_html) if qc_html else output_dir / "qc.html"
    build_contact_sheet(records, qc_path, detector_name=det.name)
    return records
