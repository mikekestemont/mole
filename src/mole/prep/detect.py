"""Text-zone detectors for `mole prep`.

Goal: given a page scan, find the bounding box of the MAIN handwritten text zone
so it can be cropped out (dropping rulers, colour charts, hands, bindings,
margins, running titles / marginalia).

Pluggable ``TextZoneDetector`` interface with two backends:

* ``HeuristicTextZoneDetector`` -- classical CV (ink-density geometry). Fast,
  CPU-only, deterministic, no heavy deps. The always-available default.
* ``YoloTextZoneDetector`` -- ``magistermilitum/YOLO_manuscripts`` (a
  manuscript-trained YOLOv11x-OBB, MIT). Semantic (knows Text vs Paratext vs
  Decoration), GPU-accelerated. Opt-in via the ``mole[detect]`` extra.

Heavy imports are lazy so ``import mole`` stays light.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

# YOLO-gen classes that constitute the MAIN text zone (exclude Paratext = marg\
# inalia/headers, Decoration, etc. — the whole point is to isolate main text).
YOLO_TEXT_CLASSES: tuple[str, ...] = ("Text", "Text_Main")
DEFAULT_YOLO_REPO = "magistermilitum/YOLO_manuscripts"
DEFAULT_YOLO_WEIGHTS = "best.pt"

BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1), axis-aligned, pixel coords


@dataclass
class Detection:
    """One detected region."""

    bbox: BBox
    label: str
    score: float
    polygon: list[tuple[float, float]] | None = None  # OBB corners, if available


@runtime_checkable
class TextZoneDetector(Protocol):
    """Anything that maps an image path to a list of :class:`Detection`."""

    name: str

    def detect(self, image_path: str | Path) -> list[Detection]: ...


def union_bbox(dets: list[Detection]) -> BBox | None:
    """Axis-aligned union of detection bounding boxes (``None`` if empty)."""
    if not dets:
        return None
    x0 = min(d.bbox[0] for d in dets)
    y0 = min(d.bbox[1] for d in dets)
    x1 = max(d.bbox[2] for d in dets)
    y1 = max(d.bbox[3] for d in dets)
    return (x0, y0, x1, y1)


def main_text_zone(dets: list[Detection], text_labels: tuple[str, ...],
                   image_size: tuple[int, int], padding: int = 0) -> BBox | None:
    """Union of text-class detections, padded and clipped to the image.

    Returns ``None`` when no text-class region was found (caller decides whether
    to fall back to the whole page).
    """
    text = [d for d in dets if d.label in text_labels]
    box = union_bbox(text)
    if box is None:
        return None
    w, h = image_size
    x0, y0, x1, y1 = box
    return (max(0, x0 - padding), max(0, y0 - padding),
            min(w, x1 + padding), min(h, y1 + padding))


# --------------------------------------------------------------------------- #
class HeuristicTextZoneDetector:
    """Classical ink-density text-block detector (no learned weights).

    Sauvola-style adaptive threshold -> ink mask -> morphological closing ->
    largest dense connected component -> bounding box. Labels the result
    ``Text_Main`` so it plugs into the same zone logic as the YOLO backend.
    """

    name = "heuristic"

    def __init__(self, close_frac: float = 0.02, min_area_frac: float = 0.02, **_ignored):
        # **_ignored swallows yolo-only kwargs (conf/device) so the two backends
        # share one call signature from the CLI.
        self.close_frac = close_frac
        self.min_area_frac = min_area_frac

    def detect(self, image_path: str | Path) -> list[Detection]:
        import numpy as np
        from PIL import Image, ImageFile
        from scipy import ndimage

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        h, w = gray.shape

        # Adaptive threshold: ink is darker than a local mean by a margin.
        win = max(15, (int(min(h, w) * 0.05) | 1))  # odd window ~5% of short side
        local_mean = ndimage.uniform_filter(gray, size=win)
        ink = gray < (local_mean - 10.0)

        # Close gaps so words/lines merge into blocks.
        k = max(3, int(min(h, w) * self.close_frac))
        ink = ndimage.binary_closing(ink, structure=np.ones((k, k)))

        labels, n = ndimage.label(ink)
        if n == 0:
            return []
        # Largest component by pixel count (background label 0 excluded).
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
        biggest = int(np.argmax(sizes)) + 1
        if sizes[biggest - 1] < self.min_area_frac * h * w:
            return []
        ys, xs = np.where(labels == biggest)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        score = float(sizes[biggest - 1] / (h * w))
        return [Detection(bbox=bbox, label="Text_Main", score=score)]


class YoloTextZoneDetector:
    """YOLO-gen (``magistermilitum/YOLO_manuscripts``) text-zone detector.

    Loads the MIT-licensed manuscript YOLOv11x-OBB with plain ``ultralytics``
    (weights fetched once from the HF hub and cached). Returns all detected
    regions; :func:`main_text_zone` filters to the text classes.
    """

    name = "yolo"

    def __init__(self, repo: str = DEFAULT_YOLO_REPO, weights: str = DEFAULT_YOLO_WEIGHTS,
                 conf: float = 0.25, device: str | None = None):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        weight_path = hf_hub_download(repo_id=repo, filename=weights)
        self.model = YOLO(weight_path)
        self.conf = conf
        self.device = device

    def detect(self, image_path: str | Path) -> list[Detection]:
        res = self.model.predict(str(image_path), conf=self.conf, verbose=False,
                                 device=self.device)
        if not res:
            return []
        r = res[0]
        names = r.names
        dets: list[Detection] = []

        obb = getattr(r, "obb", None)
        if obb is not None and len(obb) > 0:
            corners = obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
            cls = obb.cls.cpu().numpy()
            conf = obb.conf.cpu().numpy()
            for pts, c, s in zip(corners, cls, conf):
                xs, ys = pts[:, 0], pts[:, 1]
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                dets.append(Detection(bbox, names[int(c)], float(s),
                                      polygon=[(float(x), float(y)) for x, y in pts]))
            return dets

        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), c, s in zip(xyxy, cls, conf):
                dets.append(Detection((int(x0), int(y0), int(x1), int(y1)),
                                      names[int(c)], float(s)))
        return dets


def get_detector(method: str = "yolo", **kwargs) -> TextZoneDetector:
    """Factory: ``method`` in ``{"yolo", "heuristic"}``."""
    method = method.lower()
    if method == "yolo":
        return YoloTextZoneDetector(**kwargs)
    if method == "heuristic":
        return HeuristicTextZoneDetector(**kwargs)
    raise ValueError(f"Unknown detector method {method!r}; choose 'yolo' or 'heuristic'.")
