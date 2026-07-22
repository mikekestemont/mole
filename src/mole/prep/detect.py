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

# YOLO-gen class FAMILIES that constitute the MAIN text zone. Matching is by
# family (the part before the first "_"), so "Text" covers {Text, Text_Main}.
# Everything else — Initial (drop capitals), Paratext (marginalia/headers),
# Decoration, Marks, Damage — stays EXCLUDED. Add a family here to include it.
ZONE_FAMILIES: tuple[str, ...] = ("Text",)
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


def _family(label: str) -> str:
    """Top-level class family, e.g. 'Text_Main' -> 'Text', 'Initial_P_DropCapital' -> 'Initial'."""
    return label.split("_", 1)[0]


# ------------------------------------------------------------- zone quality
def box_iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two axis-aligned boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def text_coverage(pred: BBox, truth: BBox) -> float:
    """Fraction of the TRUE text box retained by ``pred``.

    THE metric for this task, because the costs are asymmetric: a zone that
    includes extra background is nearly free (the contrast foreground filter
    discards blank parchment anyway), while a zone that clips text destroys
    writer signal that no downstream stage can recover. Optimise coverage first
    and tightness second — never the reverse, and never IoU alone, which
    averages the two failure modes into one number and hides which occurred.
    """
    ix0, iy0 = max(pred[0], truth[0]), max(pred[1], truth[1])
    ix1, iy1 = min(pred[2], truth[2]), min(pred[3], truth[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_t = max(0, truth[2] - truth[0]) * max(0, truth[3] - truth[1])
    return inter / area_t if area_t else 0.0


def excess_area(pred: BBox, truth: BBox) -> float:
    """Predicted area as a multiple of the true area (1.0 = perfectly tight)."""
    area_p = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
    area_t = max(0, truth[2] - truth[0]) * max(0, truth[3] - truth[1])
    return area_p / area_t if area_t else 0.0


def pad_bbox(b: BBox, padding: int, width: int = 0, height: int = 0) -> BBox:
    """Grow a box by ``padding`` px, clamped to the image when size is known."""
    x0, y0, x1, y1 = b
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1 = min(width, x1 + padding) if width else x1 + padding
    y1 = min(height, y1 + padding) if height else y1 + padding
    return (x0, y0, x1, y1)


def main_text_zone(dets: list[Detection], families: tuple[str, ...] = ZONE_FAMILIES,
                   image_size: tuple[int, int] = (0, 0), padding: int = 0) -> BBox | None:
    """Union of detections whose class FAMILY is in ``families``, padded + clipped.

    Returns ``None`` when no in-family region was found (caller decides whether to
    fall back to the whole page).
    """
    # Case-insensitive: class-name casing is a convention of whoever trained the
    # weights, not semantics. A detector fine-tuned locally may well emit "text"
    # where YOLO_manuscripts emits "Text", and a case-sensitive match would drop
    # every detection and silently fall back to whole-page on every image.
    wanted = {f.lower() for f in families}
    text = [d for d in dets if _family(d.label).lower() in wanted]
    box = union_bbox(text)
    if box is None:
        return None
    # image_size 0 means "unknown", so don't clamp — clamping to 0 would collapse
    # the box to (x0, y0, 0, 0). prep_folder always passes img.size; anything that
    # doesn't (a scoring harness, a notebook) used to get a silently zeroed zone.
    w, h = image_size
    x0, y0, x1, y1 = box
    return pad_bbox(box, padding, w, h)


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
        from scipy import ndimage

        from mole.data.patches import load_rgb

        gray = np.asarray(load_rgb(image_path).convert("L"), dtype=np.float32)
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
        # Silence OpenCV/libtiff "unknown field" warnings from proprietary camera
        # TIFF tags (harmless: pixels read fine). OpenCV reads this env var when it
        # is first imported, so set it BEFORE importing ultralytics (which pulls cv2).
        import os

        os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        # Quiet the "unauthenticated requests to the HF Hub" info line (harmless:
        # the weights are public and cached after the first download).
        try:
            from huggingface_hub.utils import logging as _hf_logging

            _hf_logging.set_verbosity_error()
        except Exception:
            pass

        # A local .pt (a detector fine-tuned on this corpus, see
        # scripts/train_zone_detector.py) short-circuits the hub download.
        local = Path(weights)
        if local.suffix == ".pt" and local.is_file():
            weight_path = str(local)
            self.model_id = f"local:{local}"
        else:
            weight_path = hf_hub_download(repo_id=repo, filename=weights)
            self.model_id = f"{repo}:{weights}"
        self.model = YOLO(weight_path)
        self.conf = conf
        self.device = device

    def detect(self, image_path: str | Path) -> list[Detection]:
        import numpy as np

        from mole.data.patches import load_rgb

        # Feed a PIL-decoded array (not the path): ultralytics' cv2 loader stacks
        # mismatched multi-frame TIFF frames and crashes, and this keeps the
        # detector's pixel space identical to mole's window sampling. cv2 wants BGR.
        rgb = np.asarray(load_rgb(image_path))
        res = self.model.predict(rgb[:, :, ::-1], conf=self.conf, verbose=False,
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
