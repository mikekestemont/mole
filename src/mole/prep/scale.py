"""Script-scale normalization: resample pages to a constant script module.

``--max-side`` normalizes a page by its *size*, not by the size of its *script*.
So a dense codex page (~30 px per line once capped) and a natively photographed
fragment (~39 px) reach the encoder at different magnifications, and the fixed
224 px window then samples a different number of words on each. The same hand,
imaged differently, drifts apart in descriptor space — which is exactly what
cross-material retrieval and a cross-corpus finetune must not do.

The fix is one resample, *after* binarization and *before* the 224 px windowing,
that makes the **script module** (px per glyph) constant everywhere::

    m_page = script module of this page      (px, see below)
    s      = m_target / m_page               (clamped)
    page  <- resample(page, s)

Measuring the module — what was tried
-------------------------------------
``profile`` (default) reads the module off the **phase-folded row-ink profile**:

1. line **pitch** = the first strong peak of the row profile's autocorrelation;
2. fold the profile at that period — averaging every text line in the page onto
   one high-SNR average line — and take the **FWHM of that average line**.

The fold is what makes it robust: the periodicity is a global property, so
noise, specks, bleed-through and a few odd blobs cannot move it, and folding
averages tens of lines together. The FWHM is the ink band's own height, so it
measures the *script* rather than the leading (unlike pitch, which mixes both).

``word`` is the connected-component recipe (horizontal RLSA to merge letters
into word blobs, median blob height). It is the more intuitive statistic, but
measured against the ground truth that matters — resample a page's grayscale by
a known factor, re-binarize, re-measure, and the estimate must move by exactly
that factor — it is not reliable on this material:

    equivariance error (IQR / median), 12 pages x 3 factors per corpus
    corpus          word blobs   profile pitch   profile body (FWHM)
    utrecht            0.87          0.001            0.042
    brackley           0.64          0.002            0.032
    flanders           0.29          0.001            0.034
    leroy-bin          0.09          0.000            0.038

Word blobs fail because their two free parameters (the RLSA kernel and the
speck/merge filters) are themselves functions of the unknown scale: iterating
the kernel from the blob heights it produces is a feedback loop that on real
pages lands anywhere between a speck field and fully merged lines. ``word`` is
kept as an opt-in for material where periodicity does not exist (a few lines of
large display script), not as a fallback — an unreliable factor is worse than
none, so a page that cannot be measured is simply left alone.

Everything here is numpy + ``scipy.ndimage`` (both already core dependencies).
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

SCALE_FILENAME = "scale.json"
METHODS = ("profile", "word")

# --- profile estimator -------------------------------------------------------
MIN_PITCH = 8               # px: a line pitch below this is noise, not writing
MAX_PITCH_FRACTION = 0.3    # the period must repeat at least ~3 times down the page
MIN_AUTOCORR = 0.12         # peak height below this = no periodic text structure
HARMONIC_TOLERANCE = 0.85   # prefer the earliest peak within this share of the best (fundamental)
FOLD_OVERSAMPLE = 4         # bins per pixel in the folded average line
FOLD_SMOOTH = 0.02          # folded profile smoothing, as a FRACTION of the period: an
                            # absolute (px) smoothing would widen a small band more than a
                            # large one, which is exactly the equivariance we must not break
PROFILE_SMOOTH = 1.0        # px: pre-smoothing for the pitch search only (kills row speckle)
BAND_QUANTILE = 0.5         # share of a line's ink whose span defines the band width
MIN_CONFIDENCE = 0.15       # min(autocorr peak, fold contrast) below this = unmeasurable

# --- word-blob estimator (opt-in; see module docstring) ----------------------
PASS1_KERNEL = 8            # px: bootstrap horizontal smoothing, before any scale is known
INTRA_WORD_FACTOR = 0.6     # kernel = 0.6 * word height (intra-word gaps scale with x-height)
MAX_PASSES = 3              # fixed-point iterations on the kernel
MIN_BLOBS = 15              # fewer word-like blobs than this = a near-blank page
MIN_BLOB_HEIGHT = 3         # px: below this a "blob" is scanner noise
SPECK_AREA_FACTOR = 0.2     # drop blobs under 0.2x the median ink area
HEIGHT_BAND = (0.4, 3.0)    # keep blobs within this band around the median height
MAX_WIDTH_FRACTION = 0.7    # a blob spanning >70% of the zone is a merged line, not a word

# --- shared ------------------------------------------------------------------
INK_COVERAGE = (0.002, 0.35)   # plausible ink share of a text page; outside = failed binarization
SCALE_CLAMP = (0.25, 4.0)      # a factor outside this says the estimate is wrong, not the page
RESIZE_EPS = 0.03              # |s-1| under the estimator's own ~4% repeatability: don't resample
DEFAULT_TARGET_MODULE = 24.0   # px: fallback target (~1.5 ViT-S/16 patches per x-height band)
DEFAULT_SAMPLE = 200           # pages sampled when measuring a corpus median

_BITONAL_FRACTION = 0.98    # share of near-0/near-255 pixels above which a page counts as binary


# ----------------------------------------------------------------- ink mask
def _intensity(img, zone: Sequence[int] | None = None):
    """``uint8`` intensity plane of a PIL image / array, optionally zone-cropped."""
    import numpy as np
    from PIL import Image

    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("L"))
    else:
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        if arr.dtype.kind == "f":
            arr = arr * 255.0 if float(arr.max(initial=0.0)) <= 1.0 else arr
        arr = arr.astype(np.uint8, copy=False)
    if zone is not None:
        h, w = arr.shape
        x0, y0, x1, y1 = (int(round(v)) for v in zone)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 >= 2 and y1 - y0 >= 2:      # ignore a degenerate zone rather than crash
            arr = arr[y0:y1, x0:x1]
    return arr


def _otsu(g) -> int:
    """Otsu's global threshold on a ``uint8`` plane (maximum between-class variance)."""
    import numpy as np

    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return 128
    levels = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * levels)
    sum_all = float((hist * levels).sum())
    ok = (w0 > 0) & (w1 > 0)
    between = np.zeros(256)
    between[ok] = ((sum_all * w0[ok] - m0[ok] * total) ** 2 / (w0[ok] * w1[ok] * total * total))
    return int(np.argmax(between))


def ink_mask(img, zone: Sequence[int] | None = None):
    """Boolean ink mask for a page (``True`` = ink), polarity auto-detected.

    Bitonal input (the normal case) is thresholded at mid-grey; anything else —
    a grayscale scan, or a binary page a resample has left anti-aliased — gets a
    global Otsu threshold. Either way ink is the **minority** tone (a page is
    never mostly ink), so black-on-white and white-on-black both work and no
    ``--invert`` bookkeeping leaks in here. Camera photos of parchment should be
    binarized first (``mole prep --binarize sauvola``): no global threshold
    survives their uneven illumination.
    """
    g = _intensity(img, zone)
    bitonal = float(((g < 16) | (g > 239)).mean()) > _BITONAL_FRACTION
    dark = g < (128 if bitonal else _otsu(g))
    return dark if float(dark.mean()) <= 0.5 else ~dark


# ------------------------------------------------------------------ estimate
@dataclass(frozen=True)
class ModuleEstimate:
    """A page's script module in px, with the evidence behind it.

    ``module`` is ``None`` when the page could not be measured (blank, failed
    binarization, no periodic text) — callers must then leave the page alone
    rather than resample it on a guess.
    """

    module: float | None
    pitch: float | None = None          # line pitch (px) — profile method only
    confidence: float = 0.0             # 0..1: min(autocorrelation peak, fold contrast)
    method: str = "profile"
    n_blobs: int = 0                    # word method only

    def __bool__(self) -> bool:
        return self.module is not None


def _row_profile(mask):
    """Ink pixels per row."""
    return mask.sum(axis=1).astype("float64")


def _line_pitch(profile) -> tuple[float | None, float]:
    """Line pitch (px) as the fundamental peak of the row profile's autocorrelation.

    Returns ``(pitch, strength)``. The autocorrelation is computed by FFT, so
    this is O(n log n) on the page height — cheaper than any pixel-level pass.
    The *earliest* peak within :data:`HARMONIC_TOLERANCE` of the strongest is
    taken, because a perfectly periodic page peaks equally at 2x and 3x the
    pitch and the fundamental is the one we want; the position is then refined
    sub-pixel by a parabola through its neighbours.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    if profile.size < 4 * MIN_PITCH or float(profile.max(initial=0.0)) <= 0:
        return None, 0.0
    q = gaussian_filter1d(profile, PROFILE_SMOOTH)
    q = q - q.mean()
    n = int(2 ** np.ceil(np.log2(q.size * 2)))
    spectrum = np.fft.rfft(q, n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), n)[:q.size]
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]
    hi = min(ac.size - 1, max(MIN_PITCH + 3, int(q.size * MAX_PITCH_FRACTION)))
    seg = ac[MIN_PITCH:hi]
    if seg.size < 3:
        return None, 0.0
    peaks = [i for i in range(1, seg.size - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return None, 0.0
    best = max(float(seg[i]) for i in peaks)
    if best < MIN_AUTOCORR:
        return None, best
    idx = next(i for i in peaks if seg[i] >= HARMONIC_TOLERANCE * best)
    lag = float(MIN_PITCH + idx)
    if 0 < idx < seg.size - 1:
        y0, y1, y2 = (float(seg[idx + d]) for d in (-1, 0, 1))
        den = y0 - 2 * y1 + y2
        if den:
            lag += float(np.clip((y0 - y2) / (2 * den), -1.0, 1.0))
    return lag, best


def _fold(profile, pitch):
    """Average every text line onto one period — the page's mean line profile.

    Folded from the *raw* profile: the pitch search smooths in pixels, which is
    fine for finding a period but would blur a narrow band more than a wide one.
    The residual smoothing here is a fraction of the period, so it treats every
    scale alike.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    bins = max(16, int(round(pitch * FOLD_OVERSAMPLE)))
    pos = (np.arange(profile.size) % pitch) * (bins / pitch)
    lo = np.floor(pos).astype(int) % bins
    frac = pos - np.floor(pos)
    acc = (np.bincount(lo, weights=profile * (1 - frac), minlength=bins)
           + np.bincount((lo + 1) % bins, weights=profile * frac, minlength=bins))
    cnt = (np.bincount(lo, weights=1 - frac, minlength=bins)
           + np.bincount((lo + 1) % bins, weights=frac, minlength=bins))
    return gaussian_filter1d(acc / np.maximum(cnt, 1e-9), max(1.0, bins * FOLD_SMOOTH),
                             mode="wrap")


def _band_width(folded, pitch, quantile: float = BAND_QUANTILE) -> tuple[float | None, float]:
    """Width of the mean line's ink band (px), and how line-like the page is (0..1).

    The band is measured as an *interquartile* width: read the folded profile as
    the distribution of ink over the phase of a line, take the span holding its
    central ``quantile`` of the ink, and rescale so that a uniform band of width
    W measures W. Integrating this way rather than cutting the profile at half
    its height (FWHM) is what makes it robust — a hand that piles ink on the
    baseline, or a ruled line inside the band, spikes the profile and halves an
    FWHM while barely moving a quantile. Measured on synthetic pages spanning an
    octave of script sizes, width/x-height held to 0.89-0.93 here against
    0.25-0.88 for the FWHM.

    Contrast (peak-to-trough over peak) comes back as evidence that the page has
    a text rhythm at all: a page without one folds into a flat profile.
    """
    import numpy as np

    f = np.roll(folded, folded.size // 2 - int(np.argmax(folded)))   # centre the peak
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return None, 0.0
    mass = f - lo
    total = float(mass.sum())
    if total <= 0:
        return None, 0.0
    cdf = np.cumsum(mass) / total
    bins = np.arange(f.size, dtype=np.float64)
    span = (np.interp(0.5 + quantile / 2, cdf, bins)
            - np.interp(0.5 - quantile / 2, cdf, bins))
    return float(span / quantile * pitch / f.size), float((hi - lo) / max(hi, 1e-9))


def _support_box(mask) -> tuple[int, int, int, int] | None:
    """Bounding box of the largest bright region — the physical support (parchment).

    Charters are often photographed on a black cloth, and then "the minority
    tone" is the cloth plus the ink, not the ink. Isolating the biggest bright
    blob recovers the leaf itself (and drops the ruler and colour chart beside
    it). Only accepted when it really is a sub-region of the page, so a page
    that simply has a lot of ink is not silently cropped.
    """
    import numpy as np
    from scipy.ndimage import find_objects, label

    lab, n = label(~mask)
    if n == 0:
        return None
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    areas[0] = 0                                   # label 0 is the ink itself
    box = find_objects(lab)[int(np.argmax(areas)) - 1]
    x0, y0 = box[1].start, box[0].start
    x1, y1 = box[1].stop, box[0].stop
    share = ((x1 - x0) * (y1 - y0)) / max(mask.size, 1)
    return (x0, y0, x1, y1) if 0.1 <= share <= 0.9 else None


def estimate_module(img, zone: Sequence[int] | None = None, *, method: str = "profile",
                    min_confidence: float = MIN_CONFIDENCE) -> ModuleEstimate:
    """Measure a page's script module (px) — see the module docstring for the method.

    ``zone`` (``x0, y0, x1, y1``, e.g. from ``zones.json``) restricts the
    measurement to the text area: margins, rulers and decoration otherwise skew
    it. Returns an estimate whose ``module`` is ``None`` when the page cannot be
    trusted to have been measured.
    """
    if method not in METHODS:
        raise ValueError(f"scale method must be one of {METHODS}, got {method!r}")
    mask = ink_mask(img, zone)
    coverage = float(mask.mean()) if mask.size else 0.0
    if coverage > INK_COVERAGE[1] and zone is None:
        support = _support_box(mask)               # a dark photographic background?
        if support is not None:
            mask = ink_mask(img, support)
            coverage = float(mask.mean()) if mask.size else 0.0
    if not INK_COVERAGE[0] <= coverage <= INK_COVERAGE[1]:
        # A blank page, or a binarization that turned the parchment into ink —
        # either way there is nothing to measure and nothing worth resampling.
        return ModuleEstimate(None, method=method)
    if method == "word":
        return _estimate_word(mask)
    profile = _row_profile(mask)
    pitch, strength = _line_pitch(profile)
    if pitch is None:
        return ModuleEstimate(None, None, strength, "profile")
    width, contrast = _band_width(_fold(profile, pitch), pitch)
    confidence = min(strength, contrast)
    if width is None or width <= 0 or confidence < min_confidence:
        return ModuleEstimate(None, pitch, confidence, "profile")
    return ModuleEstimate(float(width), float(pitch), float(confidence), "profile")


def script_module(img, zone: Sequence[int] | None = None, **kwargs) -> float | None:
    """Convenience wrapper: :func:`estimate_module` reduced to its module in px."""
    return estimate_module(img, zone, **kwargs).module


# -------------------------------------------------------- word-blob estimator
def _close_horizontal(mask, kernel: int):
    """Merge letters into words: a horizontal closing with a ``kernel``-wide element.

    Closing (dilate then erode) bridges intra-word gaps without inflating the
    blob's outer extent, so heights stay measurable — unlike a plain dilation,
    which fattens every blob. Two 1-D running filters, so the cost does not grow
    with the kernel width.
    """
    import numpy as np
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    if kernel <= 1:
        return mask
    u = mask.astype(np.uint8)
    u = maximum_filter1d(u, kernel, axis=1, mode="constant", cval=0)
    u = minimum_filter1d(u, kernel, axis=1, mode="constant", cval=1)  # cval=1: don't eat the edges
    return u.astype(bool)


def _blob_metrics(mask, kernel: int):
    """``(heights, widths, areas)`` of the connected components of the closed mask."""
    import numpy as np
    from scipy.ndimage import find_objects, label

    lab, n = label(_close_horizontal(mask, kernel), structure=np.ones((3, 3), np.uint8))
    if n == 0:
        z = np.zeros(0)
        return z, z, z
    boxes = find_objects(lab)
    heights = np.array([b[0].stop - b[0].start for b in boxes], dtype=np.float64)
    widths = np.array([b[1].stop - b[1].start for b in boxes], dtype=np.float64)
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    return heights, widths, areas


def _word_like(heights, widths, areas, zone_width: int):
    """Heights of the blobs that plausibly are words.

    Drops noise specks, decorated initials / multi-line merges (far taller than
    the median), and — only when enough narrow blobs remain — anything spanning
    most of the zone width, which is a merged line or a ruled border rather than
    a word. "Prefer, don't demand": discarding merged lines unconditionally is
    what makes the kernel loop collapse onto the leftover junk.
    """
    import numpy as np

    keep = heights >= MIN_BLOB_HEIGHT
    if not keep.any():
        return np.zeros(0)
    keep &= areas >= SPECK_AREA_FACTOR * float(np.median(areas[keep]))
    if not keep.any():
        return np.zeros(0)
    med_h = float(np.median(heights[keep]))
    lo, hi = HEIGHT_BAND
    keep &= (heights >= lo * med_h) & (heights <= hi * med_h)
    narrow = keep & (widths <= MAX_WIDTH_FRACTION * max(zone_width, 1))
    return heights[narrow] if int(narrow.sum()) >= MIN_BLOBS else heights[keep]


def _estimate_word(mask) -> ModuleEstimate:
    """The plan's original recipe: adaptive RLSA + median word-blob height."""
    import numpy as np

    kernel, module, n_blobs = PASS1_KERNEL, None, 0
    for _ in range(MAX_PASSES):
        hs = _word_like(*_blob_metrics(mask, kernel), mask.shape[1])
        if len(hs) == 0:
            break
        module, n_blobs = float(np.median(hs)), int(len(hs))
        nxt = max(3, int(round(INTRA_WORD_FACTOR * module)))
        if nxt == kernel:                       # fixed point: same kernel, same blobs
            break
        kernel = nxt
    if module is None or n_blobs < MIN_BLOBS:
        return ModuleEstimate(None, method="word", n_blobs=n_blobs)
    return ModuleEstimate(module, method="word", n_blobs=n_blobs, confidence=1.0)


# ------------------------------------------------------------------ resample
def scale_factor(module: float | None, target: float,
                 clamp: tuple[float, float] = SCALE_CLAMP) -> float:
    """Resampling factor taking a page's module to ``target``, clamped.

    A factor outside the clamp means the *estimate* is implausible (a page of
    marginalia, a mis-detected zone), not that the page really needs a 6x
    resample — so it is clipped rather than trusted.
    """
    if not module or module <= 0 or target <= 0:
        return 1.0
    return float(min(max(target / float(module), clamp[0]), clamp[1]))


def resample(img, factor: float, *, rethreshold: bool = False):
    """Resample a PIL image by ``factor`` (LANCZOS down, BICUBIC up).

    Prefer resampling the **grayscale** original and re-binarizing afterwards
    (what ``mole prep`` does): resampling a bitonal image aliases its strokes.
    ``rethreshold`` re-binarizes the result for the case where only the binary
    exists.
    """
    from PIL import Image

    if abs(factor - 1.0) < RESIZE_EPS:
        return img
    w, h = img.size
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    out = img.resize(size, Image.LANCZOS if factor < 1.0 else Image.BICUBIC)
    if rethreshold:
        import numpy as np

        a = np.asarray(out.convert("L"))
        out = Image.fromarray(np.where(a > 128, 255, 0).astype(np.uint8), mode="L")
    return out


def scaled_zone(zone: Sequence[int] | None, factor: float) -> tuple[int, ...] | None:
    """Carry a zone bbox through a resample (its coordinates are page pixels)."""
    if zone is None:
        return None
    return tuple(int(round(v * factor)) for v in zone)


# ------------------------------------------------------------------ manifest
@dataclass
class ScaleEntry:
    """What normalization did to one page (provenance + reuse)."""

    module: float | None            # module measured BEFORE resampling
    scale: float                    # factor applied
    size: tuple[int, int]           # (w, h) of the image actually written
    pitch: float | None = None
    confidence: float = 0.0
    module_out: float | None = None  # module measured AFTER — the collapse evidence

    @property
    def current_module(self) -> float | None:
        """The page's module as it now sits on disk (measured, else predicted)."""
        if self.module_out is not None:
            return self.module_out
        return self.module * self.scale if self.module else None


@dataclass
class ScaleManifest:
    """``scale.json``: the target a folder was normalized to, and per-page factors."""

    meta: dict[str, Any] = field(default_factory=dict)
    images: dict[str, ScaleEntry] = field(default_factory=dict)

    @property
    def target(self) -> float | None:
        t = self.meta.get("script_module_target")
        return float(t) if t else None

    def module_for(self, name: str, size: tuple[int, int] | None = None) -> float | None:
        """The page's current module, or ``None`` if unknown or stale.

        ``size`` guards against a manifest left behind by another run: if the
        recorded size does not match the image on disk the entry is ignored and
        the caller re-measures.
        """
        entry = self.images.get(name)
        if entry is None or (size is not None and tuple(entry.size) != tuple(size)):
            return None
        return entry.current_module


def save_scale(path: str | Path, manifest: ScaleManifest) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": manifest.meta,
        "images": {
            name: {"module": e.module, "scale": e.scale, "size": list(e.size),
                   "pitch": e.pitch, "confidence": e.confidence, "module_out": e.module_out}
            for name, e in manifest.images.items()
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_scale(path: str | Path) -> ScaleManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    images = {
        name: ScaleEntry(module=v.get("module"), scale=float(v.get("scale", 1.0)),
                         size=tuple(v.get("size", (0, 0))), pitch=v.get("pitch"),
                         confidence=float(v.get("confidence", 0.0)),
                         module_out=v.get("module_out"))
        for name, v in data.get("images", {}).items()
    }
    return ScaleManifest(meta=data.get("meta", {}), images=images)


def find_scale(dataset_root: str | Path) -> Path | None:
    """Return the dataset's ``scale.json`` if present (auto-discovery, like zones)."""
    p = Path(dataset_root) / SCALE_FILENAME
    return p if p.is_file() else None


def scale_meta(target: float, method: str, source: str, **extra: Any) -> dict[str, Any]:
    """Standard ``scale.json`` meta block."""
    return {"method": method, "script_module_target": float(target), "target_source": source,
            "clamp": list(SCALE_CLAMP), "created": _dt.datetime.now().isoformat(timespec="seconds"),
            **extra}


# -------------------------------------------------------------- page scaler
@dataclass(frozen=True)
class Rescaled:
    """Result of scaling one page: the image, plus what was done and on what basis."""

    image: Any
    factor: float
    module: float | None
    source: str          # "manifest" | "measured" | "unmeasurable"

    @property
    def changed(self) -> bool:
        return abs(self.factor - 1.0) >= RESIZE_EPS


class PageScaler:
    """Resamples pages to a fixed script module — the inference-side normalizer.

    It scales by ``target / current module``, so running it on an already
    normalized folder is a no-op up to the estimator's own accuracy (a few
    percent): ``mole prep`` records the *post*-normalization module in
    ``scale.json``, so the residual factor is ~1 and — being below
    ``RESIZE_EPS`` — no pixels are touched and nothing is re-measured. A folder
    that was never normalized is measured page by page instead. Either way the
    pages that reach the encoder share one module, which is what keeps them in
    the regime the VLAD codebook learned its vocabulary in.
    """

    def __init__(self, target: float, *, method: str = "profile",
                 manifest: ScaleManifest | None = None,
                 clamp: tuple[float, float] = SCALE_CLAMP, rethreshold: bool = False):
        if target <= 0:
            raise ValueError(f"target script module must be positive, got {target!r}")
        if method not in METHODS:
            raise ValueError(f"scale method must be one of {METHODS}, got {method!r}")
        self.target = float(target)
        self.method = method
        self.manifest = manifest
        self.clamp = clamp
        self.rethreshold = rethreshold
        self.factors: list[float] = []
        self.modules: list[float] = []
        self.n_pages = self.n_rescaled = self.n_unmeasurable = self.n_from_manifest = 0

    def rescale(self, page, *, name: str | None = None,
                zone: Sequence[int] | None = None) -> Rescaled:
        """Return ``page`` resampled so its script module is ``target`` px."""
        self.n_pages += 1
        module, source = None, "measured"
        if name and self.manifest is not None:
            module = self.manifest.module_for(name, size=page.size)
            if module is not None:
                source, self.n_from_manifest = "manifest", self.n_from_manifest + 1
        if module is None:
            module = estimate_module(page, zone, method=self.method).module
        if module is None:
            self.n_unmeasurable += 1
            return Rescaled(page, 1.0, None, "unmeasurable")
        factor = scale_factor(module, self.target, self.clamp)
        self.modules.append(float(module))
        self.factors.append(factor)
        out = Rescaled(resample(page, factor, rethreshold=self.rethreshold), factor,
                       float(module), source)
        if out.changed:
            self.n_rescaled += 1
        return out

    def summary(self) -> dict[str, Any]:
        """Provenance block for an embedding sidecar."""
        import numpy as np

        med = float(np.median(self.factors)) if self.factors else 1.0
        med_m = float(np.median(self.modules)) if self.modules else None
        return {"script_module_target": self.target, "method": self.method,
                "pages": self.n_pages, "rescaled": self.n_rescaled,
                "from_manifest": self.n_from_manifest, "unmeasurable": self.n_unmeasurable,
                "median_scale": round(med, 4),
                "median_module": round(med_m, 2) if med_m else None}

    def note(self) -> str:
        """One-line human summary for the console."""
        s = self.summary()
        src = f", {s['from_manifest']} from scale.json" if s["from_manifest"] else ""
        miss = f", {s['unmeasurable']} unmeasurable" if s["unmeasurable"] else ""
        return (f"scale: {s['rescaled']}/{s['pages']} pages resampled to module "
                f"{self.target:g}px (median ×{s['median_scale']:g}{src}{miss})")


# ----------------------------------------------------------- corpus measure
@dataclass(frozen=True)
class CorpusScale:
    """Script module of a whole folder: the median page, and the spread around it."""

    median: float | None
    modules: dict[str, float]
    n_measured: int
    n_failed: int
    directory: str = ""
    method: str = "profile"

    @property
    def quartiles(self) -> tuple[float, float] | None:
        import numpy as np

        if not self.modules:
            return None
        v = np.fromiter(self.modules.values(), dtype=float)
        return float(np.percentile(v, 25)), float(np.percentile(v, 75))

    @property
    def spread(self) -> float | None:
        """IQR as a fraction of the median — how scale-uniform this folder is."""
        q = self.quartiles
        return None if (q is None or not self.median) else (q[1] - q[0]) / self.median


def _evenly_spaced(items: Sequence, n: int | None) -> list:
    """A deterministic, order-preserving subsample (no RNG, covers the whole folder)."""
    if not n or n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _is_bitonal(img) -> bool:
    import numpy as np

    g = np.asarray(img.convert("L")) if hasattr(img, "convert") else np.asarray(img)
    return float(((g < 16) | (g > 239)).mean()) > _BITONAL_FRACTION


def measure_corpus(directory: str | Path, *, sample: int | None = DEFAULT_SAMPLE,
                   method: str = "profile", binarize: str = "auto",
                   max_side: int | None = None, sauvola_window: int = 25,
                   sauvola_k: float = 0.2, use_zones: bool = True,
                   reuse_manifest: bool = True, progress: bool = True) -> CorpusScale:
    """Median script module over (a sample of) a folder of pages.

    This is how the **target** is chosen: run it on the corpus the active VLAD
    codebook was fitted on, and every other corpus is then resampled to that
    module before tiling, so all descriptors land in the regime the vocabulary
    was learned in. Run it again on a normalized folder to verify the collapse.

    ``binarize="auto"`` binarizes only pages that are not already bitonal, so it
    works on a raw and on a ``-bin`` folder alike. Sampling is deterministic and
    evenly spaced; a median needs far fewer pages than the whole corpus.
    """
    import numpy as np

    from mole.data.datasets import IMAGE_EXTENSIONS
    from mole.data.patches import load_rgb
    from mole.data.zones import find_zones, load_zones
    from mole.prep.binarize import binarize_image, downscale_max_side
    from mole.progress import track

    directory = Path(directory)
    files = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"no images in {directory}")
    files = _evenly_spaced(files, sample)

    zpath = find_zones(directory) if use_zones else None
    zones = load_zones(zpath) if zpath else None
    spath = find_scale(directory) if reuse_manifest else None
    manifest = load_scale(spath) if spath else None

    modules: dict[str, float] = {}
    n_failed = 0
    it = track(files, f"Measuring {directory.name}", unit="page") if progress else files
    for f in it:
        page = downscale_max_side(load_rgb(f), max_side)
        if binarize == "sauvola" or (binarize == "auto" and not _is_bitonal(page)):
            from mole.prep.stretch import bbox_mask
            zone = zone_for(zones, f.name, page.size)
            page = binarize_image(
                page, window=sauvola_window, k=sauvola_k,
                stretch_mask=bbox_mask(page.size[::-1], zone),
            )
        value = manifest.module_for(f.name, size=page.size) if manifest else None
        if value is None:
            value = estimate_module(page, zone_for(zones, f.name, page.size),
                                    method=method).module
        if value is None:
            n_failed += 1
        else:
            modules[f.name] = float(value)

    median = float(np.median(list(modules.values()))) if modules else None
    return CorpusScale(median=median, modules=modules, n_measured=len(modules),
                       n_failed=n_failed, directory=str(directory), method=method)


def zone_for(zones, name: str, size: tuple[int, int]):
    """Text-zone bbox for a page, carried to ``size`` (zones.json holds pre-cap coords)."""
    if zones is None:
        return None
    bbox = zones.bbox_for(name)
    entry = zones.images.get(name)
    if bbox is None:
        return None
    ratio = (size[0] / entry.size[0]) if (entry and entry.size and entry.size[0]) else 1.0
    return scaled_zone(bbox, ratio)


def corpus_target(directories: Iterable[str | Path],
                  **kwargs) -> tuple[float | None, list[CorpusScale]]:
    """Pooled target across several folders: the median of their page medians.

    One vote per folder — a median over pooled pages would let the largest
    collection set the target for everyone.
    """
    import numpy as np

    scales = [measure_corpus(d, **kwargs) for d in directories]
    medians = [s.median for s in scales if s.median]
    return (float(np.median(medians)) if medians else None), scales
