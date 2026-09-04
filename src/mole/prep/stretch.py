"""Robust percentile contrast stretch for grayscale pages.

Camera photos and microfilm vary in exposure far more than in the hand: a
washed plate sits in a 40-grey-level band, a dark mount fills the histogram
with zeros. Mapping the interior 2nd–98th percentiles onto a fixed ink/paper
range (default p2→20, p98→255) is a per-page linear fit — no model, O(pixels)
— so it travels across archives.

Percentiles are estimated on the text zone when a bbox is given, so black
mounts and white AABB fill do not set the range. The linear map is then
applied to the whole page. Already-bitonal images are left alone.

This runs *before* Sauvola: stretching a 0/255 bitmap is a no-op, and faint
ink that would vanish under a local threshold is first pulled away from the
parchment.
"""

from __future__ import annotations

P_LO, P_HI = 2.0, 98.0
OUT_LO, OUT_HI = 20, 255
MIN_SPAN = 8
MIN_PIXELS = 100


def bbox_mask(shape_hw: tuple[int, int], bbox: tuple[int, int, int, int] | None):
    """Boolean mask of ``bbox`` (x0, y0, x1, y1) on an array of shape ``(h, w)``."""
    import numpy as np

    if bbox is None:
        return None
    h, w = shape_hw
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def is_bitonal(gray) -> bool:
    """True when the page is already two-level *at the histogram ends*.

    A washed plate with only two mid-gray values is not bitonal — that is
    the case the stretch is for. ``unique <= 2`` alone would skip it.
    JPEG-saved bitonal still clusters at 0/255, so the 98% end-mass test
    matches :func:`mole.prep.scale._is_bitonal`.
    """
    import numpy as np

    g = np.asarray(gray)
    if g.size == 0:
        return True
    uniq = np.unique(g)
    if uniq.size <= 2 and np.all((uniq < 16) | (uniq > 239)):
        return True
    return float(((g < 16) | (g > 239)).mean()) > 0.98


def stretch_gray(gray, mask=None, *, p_lo: float = P_LO, p_hi: float = P_HI,
                 out_lo: int = OUT_LO, out_hi: int = OUT_HI,
                 min_span: float = MIN_SPAN):
    """Map interior percentiles onto ``[out_lo, out_hi]``.

    Returns ``(stretched uint8 array, stats dict)``. ``stats["applied"]`` is 0
    when the page was skipped (bitonal, too few pixels, or span below
    ``min_span``).
    """
    import numpy as np

    arr = np.asarray(gray)
    if arr.ndim == 3:
        # RGB → luminance; caller usually converts first
        arr = np.round(0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    work = arr if arr.dtype == np.uint8 else np.clip(np.round(arr), 0, 255).astype(np.uint8)
    stats = {"p2": None, "p98": None, "span": None, "n": 0, "applied": 0}
    if is_bitonal(work):
        return work.copy(), stats
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != work.shape:
            raise ValueError(
                f"stretch mask shape {mask.shape} does not match gray {work.shape}")
        pix = work[mask]
    else:
        pix = work.reshape(-1)
    n = int(pix.size)
    stats["n"] = n
    if n < MIN_PIXELS:
        return work.copy(), stats
    lo, hi = np.percentile(pix.astype(np.float32), [p_lo, p_hi])
    span = float(hi - lo)
    stats.update(p2=float(lo), p98=float(hi), span=span)
    if span < min_span:
        return work.copy(), stats
    scale = (out_hi - out_lo) / span
    mapped = np.clip(np.round(out_lo + (work.astype(np.float32) - lo) * scale), 0, 255)
    out = mapped.astype(np.uint8)
    stats["applied"] = 1
    return out, stats
