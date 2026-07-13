"""Optional preprocessing: main-text-zone isolation + QC.

This is an OPTIONAL upstream stage. Pipeline::

    raw page -> [mole prep: text-zone crop] -> patch-window sampling -> ...

``mole prep`` writes a new folder of cropped pages (+ a QC contact sheet) that
train/embed then consume like any dataset.

Detectors (pluggable):

* ``heuristic`` -- classical ink-density CV; no learned weights, CPU-only.
* ``yolo``      -- ``magistermilitum/YOLO_manuscripts`` (MIT YOLOv11x-OBB),
  opt-in via the ``mole[detect]`` extra.
"""

from __future__ import annotations

from mole.prep.detect import (Detection, HeuristicTextZoneDetector,
                              TextZoneDetector, YoloTextZoneDetector, get_detector)
from mole.prep.run import PrepRecord, prep_folder

__all__ = [
    "Detection", "TextZoneDetector", "HeuristicTextZoneDetector",
    "YoloTextZoneDetector", "get_detector", "PrepRecord", "prep_folder",
]
