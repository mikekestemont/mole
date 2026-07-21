"""Charter images for the review sheet: whole pages, losslessly, inline.

Three findings shape this module, all measured on ``data/leroy-bin`` rather than
assumed (see ``REVIEW_PLAN.md`` §3):

* **Page thumbnails are useless here.** At ~5% of native scale you cannot judge
  letterforms, which is the entire point of the exercise.
* **Lossy compression is the wrong tool for binarized pages.** Sharp black/white
  edges are pure high-frequency content — a 1200px JPEG crop costs 90–220 KB while
  the *whole page* as lossless WebP costs 38–49 KB. Lossy is 3–5× BIGGER.
* **So no cropping is needed.** A whole page at native resolution, losslessly
  encoded, is affordable — which also removes the risk of a crop-picker framing a
  seal instead of running text.

Colour pages are binarized on the fly (the Sauvola path from :mod:`mole.prep`),
because bilevel is both far smaller and closer to what the model sees.

Everything is inlined as a data URI: the finished report is ONE file with no
sidecar directory and no relative paths to break when it is emailed. The optional
on-disk cache is a *build-time* artifact that never travels with the report.
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

import numpy as np

# encode order: WebP lossless is smallest for bilevel, PNG is the fallback
_FORMATS = (("WEBP", {"lossless": True, "quality": 100, "method": 4}),
            ("PNG", {"optimize": True}))


def _is_bilevel(img) -> bool:
    a = np.asarray(img.convert("L"))
    return len(np.unique(a)) <= 2


def encode_page(path: str | Path, *, max_width: int = 1600,
                binarize: bool = True) -> tuple[bytes, str]:
    """Encode one page as compact lossless bytes; returns ``(data, mime)``.

    Downscaling is capped rather than aggressive: ``max_width`` only bites on
    scans wider than it, because resolution is exactly what makes the script
    legible.
    """
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    img = Image.open(path)
    if getattr(img, "n_frames", 1) > 1:            # multi-frame TIFF: largest frame
        best, area = 0, 0
        for i in range(img.n_frames):
            img.seek(i)
            if img.size[0] * img.size[1] > area:
                best, area = i, img.size[0] * img.size[1]
        img.seek(best)
    img = img.convert("L")

    if binarize and not _is_bilevel(img):
        from mole.prep.binarize import binarize_image

        img = binarize_image(img).convert("L")

    if img.width > max_width:
        h = max(1, int(img.height * max_width / img.width))
        img = img.resize((max_width, h))
    img = img.convert("1") if _is_bilevel(img) else img

    best_blob, best_mime = None, "image/png"
    for fmt, kw in _FORMATS:
        try:
            buf = io.BytesIO()
            img.save(buf, fmt, **kw)
        except (OSError, KeyError, ValueError):
            continue                                # e.g. Pillow built without WebP
        blob = buf.getvalue()
        if best_blob is None or len(blob) < len(best_blob):
            best_blob, best_mime = blob, f"image/{fmt.lower()}"
    if best_blob is None:                           # nothing encoded: give up cleanly
        raise RuntimeError(f"could not encode {path}")
    return best_blob, best_mime


def data_uri(blob: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(blob).decode("ascii")


class ImageBudget:
    """Encode pages in priority order until a byte budget is exhausted.

    The report has to survive an email attachment limit, so the cap is enforced
    at build time and *reported*, never silently exceeded. Documents are supplied
    most-important-first (the top of each suggestion list); whatever does not fit
    simply has no image, and the UI falls back to the filename.

    ``cache_dir`` persists encoded bytes between builds keyed by path, mtime and
    width. It is a build-time convenience only — the finished HTML embeds the
    bytes and has no dependency on it.
    """

    def __init__(self, max_bytes: int, *, max_width: int = 1600,
                 cache_dir: str | Path | None = None, binarize: bool = True):
        self.max_bytes = max_bytes
        self.max_width = max_width
        self.binarize = binarize
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.used = 0
        self.uris: dict[str, str] = {}
        self.skipped = 0
        self.failed = 0

    def _cache_path(self, path: Path) -> Path | None:
        if not self.cache_dir:
            return None
        try:
            stamp = f"{path.resolve()}|{path.stat().st_mtime_ns}|{self.max_width}"
        except OSError:
            return None
        return self.cache_dir / (hashlib.sha256(stamp.encode()).hexdigest()[:16] + ".bin")

    def add(self, key: str, path: str | Path) -> bool:
        """Encode ``path`` under ``key`` if it fits the budget. Returns success."""
        if key in self.uris:
            return True
        path = Path(path)
        cp = self._cache_path(path)
        blob = mime = None
        if cp is not None and cp.is_file():
            raw = cp.read_bytes()
            sep = raw.index(b"\0")
            mime, blob = raw[:sep].decode(), raw[sep + 1:]
        else:
            if not path.is_file():
                self.failed += 1
                return False
            try:
                blob, mime = encode_page(path, max_width=self.max_width,
                                         binarize=self.binarize)
            except Exception:                       # a broken scan must not kill the report
                self.failed += 1
                return False
            if cp is not None:
                cp.write_bytes(mime.encode() + b"\0" + blob)

        cost = int(len(blob) * 4 / 3) + 64          # base64 inflation + the attribute
        if self.max_bytes and self.used + cost > self.max_bytes:
            self.skipped += 1
            return False
        self.used += cost
        self.uris[key] = data_uri(blob, mime)
        return True

    def summary(self) -> str:
        mb = self.used / (1024 * 1024)
        out = f"{len(self.uris)} page images, {mb:.1f} MB inline"
        if self.skipped:
            out += f" ({self.skipped} omitted to stay under the size cap)"
        if self.failed:
            out += f" ({self.failed} could not be read)"
        return out
