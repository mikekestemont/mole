"""Patch-window sampling from page images.

The training unit is a square *window* lifted from a page (sliding grid with
overlap), NOT the whole page. This is preserved from the original code. An
optional foreground filter drops near-empty windows.

Resolution contract (see also :mod:`mole.config`):

* ``window_size`` (default 256 px) -- physical crop size taken from the page.
* ``model_size``  (default 224 px) -- what the ViT ingests.

Training resizes window -> model_size via random-resized-crop; embedding resizes
window -> model_size deterministically. The two paths share the same
``window_size`` default so train and inference see the same distribution.

The loader normalizes every image to 3-channel internally (grayscale replicated),
so color, grayscale, and bitonal corpora all work with no user preprocessing.

Heavy imports (PIL/numpy) are lazy so ``import mole`` stays light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

# Locked in Phase 2 after visual review on medieval charters: 512px windows give
# ~4-6 words / 3-4 lines of context per sample (256 was too zoomed for writer
# style; it suited binarized ICDAR data, not these scans).
DEFAULT_WINDOW_SIZE = 512
DEFAULT_MODEL_SIZE = 224
DEFAULT_OVERLAP = 0.5


class Window(NamedTuple):
    """A window crop location: top-left ``(x, y)`` and its ``size`` in pixels."""

    x: int
    y: int
    size: int


def load_rgb(image_path: str | Path, invert: bool = False):
    """Open an image and normalize to a 3-channel RGB PIL image.

    Grayscale/bitonal inputs are replicated to 3 channels transparently. Truncated
    files are tolerated (common with mass-digitized material). ``invert`` negates
    intensity (e.g. white-on-black binarizations -> conventional black-on-white),
    which is what the foreground filter and light-background augs assume.
    """
    from PIL import Image, ImageFile, ImageOps

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(image_path)
    if getattr(img, "n_frames", 1) > 1:      # multi-frame / pyramidal TIFF
        best, area = 0, -1                   # keep the largest frame (the full image,
        for i in range(img.n_frames):        # not a thumbnail); mismatched frames also
            img.seek(i)                      # crash OpenCV's loader in the YOLO detector
            a = img.size[0] * img.size[1]
            if a > area:
                area, best = a, i
        img.seek(best)
    img = img.convert("RGB")
    return ImageOps.invert(img) if invert else img


# Foreground = INKED (high local contrast), not "dark". Text strokes create local
# intensity variance; blank parchment/paper/binarized background is smooth. This is
# polarity-invariant (std(x)==std(1-x), so black-on-white and white-on-black behave
# identically) and background-colour-agnostic (works on parchment, colour, bitonal) —
# unlike a "darker than X" test, which assumes black-ink-on-white and mistakes
# parchment for ink. The same std criterion is used at token level by
# :func:`patch_contrast_mask` (embedding / projector), so training and inference agree.
DEFAULT_CONTRAST_THRESHOLD = 0.05


def window_foreground_fractions(img, windows, contrast_threshold: float = DEFAULT_CONTRAST_THRESHOLD,
                                block: int = 8) -> list[float]:
    """Inked-pixel fraction for each window, via a single image-level contrast map.

    Computes the local std once over the whole page (two box filters), thresholds it
    into an inked mask, then reads each window's mean as a fast slice — O(1) per window
    instead of re-filtering every crop. Returns one fraction in ``[0, 1]`` per window.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    mean = uniform_filter(g, block)
    var = np.clip(uniform_filter(g * g, block) - mean * mean, 0.0, None)
    inked = var > (contrast_threshold * contrast_threshold)      # std > thr  <=>  var > thr^2
    out = []
    for w in windows:
        sub = inked[w.y:w.y + w.size, w.x:w.x + w.size]
        out.append(float(sub.mean()) if sub.size else 0.0)
    return out


def patch_contrast_mask(x, patch_size: int, threshold: float = DEFAULT_CONTRAST_THRESHOLD):
    """Per-patch inked mask for a batch of crops ``x`` ``[N, C, S, S]`` in ``[0, 1]``.

    Returns a bool tensor ``[N, num_patches]`` (row-major, matching ViT patch-token
    order): True where the patch's local std exceeds ``threshold`` (inked), False on
    smooth/blank patches. Polarity-invariant; shared by embedding, the projector and
    (via :func:`window_foreground_fractions`) training-window selection.
    """
    import torch.nn.functional as F

    g = x[:, :1]                                                 # intensity channel
    mean = F.avg_pool2d(g, patch_size).flatten(1)
    sq = F.avg_pool2d(g * g, patch_size).flatten(1)
    std = (sq - mean * mean).clamp(min=0).sqrt()
    return std > threshold


def window_coords(width: int, height: int, window_size: int = DEFAULT_WINDOW_SIZE,
                  overlap: float = DEFAULT_OVERLAP,
                  bounds: tuple[int, int, int, int] | None = None) -> list[Window]:
    """Pure-geometry grid of window locations — no image IO.

    Given only the image ``(width, height)`` (and optional ``bounds`` text zone),
    return the sliding-grid window origins. Lets datasets precompute windows from
    stored sizes (zones.json) without loading pixels.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    x0, y0, x1, y1 = (0, 0, width, height) if bounds is None else bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    stride = max(1, round(window_size * (1.0 - overlap)))

    def axis_starts(origin: int, extent: int) -> list[int]:
        if extent <= window_size:
            return [origin]
        starts = list(range(origin, origin + extent - window_size + 1, stride))
        return starts or [origin]

    xs, ys = axis_starts(x0, x1 - x0), axis_starts(y0, y1 - y0)
    return [Window(x, y, window_size) for y in ys for x in xs]


def sample_windows(image_path: str | Path, window_size: int = DEFAULT_WINDOW_SIZE,
                   overlap: float = DEFAULT_OVERLAP, foreground_min: float = 0.0,
                   bounds: tuple[int, int, int, int] | None = None) -> list[Window]:
    """Return window crop locations for a single page image.

    Wraps :func:`window_coords` and, when ``foreground_min > 0``, loads the image
    to drop windows whose foreground (ink) fraction is below the threshold.
    ``bounds`` restricts sampling to the prep text zone.
    """
    img = load_rgb(image_path)
    w, h = img.size
    coords = window_coords(w, h, window_size, overlap, bounds)
    if foreground_min <= 0.0:
        return coords
    fractions = window_foreground_fractions(img, coords)
    return [win for win, frac in zip(coords, fractions) if frac >= foreground_min]


def iter_window_crops(image_path: str | Path, window_size: int = DEFAULT_WINDOW_SIZE,
                      overlap: float = DEFAULT_OVERLAP, foreground_min: float = 0.0,
                      bounds: tuple[int, int, int, int] | None = None) -> Iterator:
    """Yield cropped PIL windows for a page (convenience over :func:`sample_windows`)."""
    img = load_rgb(image_path)
    for win in sample_windows(image_path, window_size, overlap, foreground_min, bounds):
        yield img.crop((win.x, win.y, win.x + win.size, win.y + win.size))
