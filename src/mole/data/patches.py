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
    img = Image.open(image_path).convert("RGB")
    return ImageOps.invert(img) if invert else img


def _foreground_fraction(crop) -> float:
    """Fraction of non-background pixels in a window (background ~ light parchment).

    Uses a simple luminance threshold: pixels darker than ~90% white count as ink.
    """
    import numpy as np

    arr = np.asarray(crop.convert("L"), dtype="float32")
    return float((arr < 0.9 * 255).mean())


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
    return [win for win in coords
            if _foreground_fraction(img.crop((win.x, win.y, win.x + win.size, win.y + win.size)))
            >= foreground_min]


def iter_window_crops(image_path: str | Path, window_size: int = DEFAULT_WINDOW_SIZE,
                      overlap: float = DEFAULT_OVERLAP, foreground_min: float = 0.0,
                      bounds: tuple[int, int, int, int] | None = None) -> Iterator:
    """Yield cropped PIL windows for a page (convenience over :func:`sample_windows`)."""
    img = load_rgb(image_path)
    for win in sample_windows(image_path, window_size, overlap, foreground_min, bounds):
        yield img.crop((win.x, win.y, win.x + win.size, win.y + win.size))
