"""Optional preprocessing: main-text-zone isolation + QC.

This is an OPTIONAL upstream stage. Pipeline::

    raw page -> [mole prep: text-zone crop] -> patch-window sampling -> ...

``mole prep`` writes a new folder of cropped pages (+ a QC contact sheet) that
train/embed then consume like any dataset.

Detectors (pluggable):

* ``heuristic`` -- classical ink-density CV; no learned weights, CPU-only.
* ``yolo``      -- ``magistermilitum/YOLO_manuscripts`` (MIT YOLOv11x-OBB),
  opt-in via the ``mole[detect]`` extra.

:mod:`mole.prep.scale` adds the second normalization axis: after binarizing,
resample every page to a constant *script* scale, so a 224 px window spans the
same amount of writing regardless of how the page was digitized.

:mod:`mole.prep.stretch` equalises *tone* first (percentile stretch on grayscale
before Sauvola). That is the default for ``mole prep --binarize sauvola``.
"""

from __future__ import annotations

from mole.prep.detect import (Detection, HeuristicTextZoneDetector,
                              TextZoneDetector, YoloTextZoneDetector, get_detector)
from mole.prep.run import PrepRecord, prep_folder, qc_from_zones
from mole.prep.scale import (CorpusScale, ModuleEstimate, PageScaler, ScaleManifest,
                             corpus_target, estimate_module, measure_corpus, resample,
                             scale_factor, script_module)
from mole.prep.stretch import bbox_mask, stretch_gray

__all__ = [
    "Detection", "TextZoneDetector", "HeuristicTextZoneDetector",
    "YoloTextZoneDetector", "get_detector", "PrepRecord", "prep_folder",
    "qc_from_zones", "CorpusScale", "ModuleEstimate", "PageScaler", "ScaleManifest",
    "corpus_target", "estimate_module", "measure_corpus", "resample", "scale_factor",
    "script_module", "bbox_mask", "stretch_gray",
]
