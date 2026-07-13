"""Augmentations tuned for premodern handwriting.

Hard rules (from the brief):

* NO horizontal or vertical flips, ever.
* Grayscale and color corpora must mix freely -> random grayscale + color jitter,
  so a color scan and a microfilm scan of similar hands land near each other.
  This is a deliberate choice for *generalization*: the network keys on stroke
  shape, not ink/parchment color, so it transfers to materials it has never seen.
* Robustness to scan quality is handled here, NOT by asking the user to convert
  files: scale/resolution jitter, blur, JPEG-compression artifacts,
  brightness/contrast. Morphological erosion/dilation (from the original code)
  simulates ink-weight variation.
* Mild rotation (+-4 deg) for scan-skew robustness (configurable; off in `mild`).

Three named presets (`mild`, `default`, `aggressive`), all overridable from the
YAML config. Values here are deliberately CALMER than the original upstream
augmentation (see ``docs`` note in the Phase-2 summary). Inspect them visually
with ``mole augview`` before committing to any training.

IMPORTANT: this module imports no heavy libraries at module top-level (so
``import mole`` / ``mole --help`` stay fast). torch / torchvision / kornia are
imported lazily inside the transform classes and functions.
"""

from __future__ import annotations

import base64
import io
import random
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path


class AugPreset(str, Enum):
    """Named augmentation strength presets."""

    MILD = "mild"
    DEFAULT = "default"
    AGGRESSIVE = "aggressive"


# Locked in Phase 2 after visual review: `mild` is the shipped training preset
# (the three names are strength labels; the chosen training strength is mild).
TRAINING_PRESET = AugPreset.MILD


@dataclass
class AugConfig:
    """All augmentation hyperparameters. Every field is overridable from YAML.

    The multi-crop contract (global + local views) is inherited from DINO/iBOT.
    """

    # --- geometry / crops ---
    model_size: int = 224          # what the ViT ingests (global views)
    local_size: int = 96           # local view size
    global_crops_scale: tuple[float, float] = (0.35, 1.0)
    local_crops_scale: tuple[float, float] = (0.10, 0.40)
    global_crops_number: int = 2
    local_crops_number: int = 6
    # --- rotation (scan-skew robustness) ---
    rotation_prob: float = 0.5
    rotation_degrees: float = 4.0
    rotation_fill: int = 255       # white ~ parchment; better than black (=ink)
    # --- color-invariance (mix color / grayscale / bitonal corpora) ---
    color_jitter_prob: float = 0.5
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2
    hue: float = 0.03
    grayscale_prob: float = 0.3
    # --- quality degradations ---
    blur_prob_global1: float = 0.3
    blur_prob_global2: float = 0.1
    blur_prob_local: float = 0.4
    jpeg_prob: float = 0.3
    jpeg_quality: tuple[int, int] = (35, 95)
    # --- ink-weight (morphological) ---
    erosion_dilation_prob: float = 0.3

    def merged(self, **overrides) -> "AugConfig":
        return replace(self, **overrides)


# Calmer-than-upstream presets. Differences vs. the original `morph` augmentation
# are documented in the Phase-2 summary. `mild` disables rotation entirely.
PRESETS: dict[str, AugConfig] = {
    AugPreset.MILD.value: AugConfig(
        global_crops_scale=(0.50, 1.0),
        local_crops_scale=(0.20, 0.50),
        local_crops_number=4,
        rotation_prob=0.0,
        color_jitter_prob=0.3, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02,
        grayscale_prob=0.2,
        blur_prob_global1=0.2, blur_prob_global2=0.1, blur_prob_local=0.3,
        jpeg_prob=0.2, jpeg_quality=(50, 95),
        erosion_dilation_prob=0.2,
    ),
    AugPreset.DEFAULT.value: AugConfig(),  # the dataclass defaults above
    AugPreset.AGGRESSIVE.value: AugConfig(
        global_crops_scale=(0.25, 1.0),
        local_crops_scale=(0.05, 0.40),
        local_crops_number=8,
        rotation_prob=0.7, rotation_degrees=4.0,
        color_jitter_prob=0.8, brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05,
        grayscale_prob=0.4,
        blur_prob_global1=0.5, blur_prob_global2=0.2, blur_prob_local=0.5,
        jpeg_prob=0.5, jpeg_quality=(25, 90),
        erosion_dilation_prob=0.5,
    ),
}


def resolve_config(preset: AugPreset | str = AugPreset.DEFAULT, **overrides) -> AugConfig:
    """Return the :class:`AugConfig` for a preset name, with optional overrides."""
    key = preset.value if isinstance(preset, AugPreset) else str(preset)
    if key not in PRESETS:
        raise ValueError(f"Unknown augmentation preset {key!r}; choose from {list(PRESETS)}")
    return PRESETS[key].merged(**overrides) if overrides else PRESETS[key]


# --------------------------------------------------------------------------- #
# Transform components (heavy imports are lazy, inside methods).
# --------------------------------------------------------------------------- #
class _RandomJPEG:
    """Simulate JPEG-compression artifacts by round-tripping through JPEG."""

    def __init__(self, p: float, quality: tuple[int, int]):
        self.p = p
        self.qmin, self.qmax = quality

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        from PIL import Image

        buf = io.BytesIO()
        q = random.randint(self.qmin, self.qmax)
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def _random_morph_kernel():
    import torch

    k = torch.rand(3, 3)
    k = (k - 0.2).round()
    k[1, 1] = 1
    return k


class _Erosion:
    def __call__(self, img):
        from kornia import morphology as morph

        return morph.erosion(img.unsqueeze(0), _random_morph_kernel())[0]


class _Dilation:
    def __call__(self, img):
        from kornia import morphology as morph

        return morph.dilation(img.unsqueeze(0), _random_morph_kernel())[0]


class MoleMultiCropAugmentation:
    """Multi-crop augmentation: 2 global (model_size) + N local (local_size) views.

    Photometric ops run on the PIL image; morphological ops run on the tensor.
    No flips. Rotation happens after the resized crop with white fill (small
    angle => negligible corner fill).
    """

    def __init__(self, cfg: AugConfig):
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.cfg = cfg
        bic = InterpolationMode.BICUBIC

        color = transforms.RandomApply(
            [transforms.ColorJitter(cfg.brightness, cfg.contrast, cfg.saturation, cfg.hue)],
            p=cfg.color_jitter_prob,
        )
        gray = transforms.RandomGrayscale(p=cfg.grayscale_prob)  # keeps 3 channels
        rotate = transforms.RandomApply(
            [transforms.RandomRotation(degrees=cfg.rotation_degrees, fill=cfg.rotation_fill)],
            p=cfg.rotation_prob,
        )
        jpeg = _RandomJPEG(cfg.jpeg_prob, cfg.jpeg_quality)

        def blur(p):
            return transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=p
            )

        photometric = [color, gray, rotate, jpeg]
        to_tensor = transforms.ToTensor()  # -> [0,1], no ImageNet normalize (matches morph)
        morph_ops = [
            transforms.RandomApply([_Erosion()], p=cfg.erosion_dilation_prob),
            transforms.RandomApply([_Dilation()], p=cfg.erosion_dilation_prob),
        ]

        self.global_transfo1 = transforms.Compose(
            [transforms.RandomResizedCrop(cfg.model_size, scale=cfg.global_crops_scale,
                                          interpolation=bic, antialias=True)]
            + photometric + [blur(cfg.blur_prob_global1), to_tensor] + morph_ops
        )
        self.global_transfo2 = transforms.Compose(
            [transforms.RandomResizedCrop(cfg.model_size, scale=cfg.global_crops_scale,
                                          interpolation=bic, antialias=True)]
            + photometric + [blur(cfg.blur_prob_global2), to_tensor] + list(reversed(morph_ops))
        )
        self.local_transfo = transforms.Compose(
            [transforms.RandomResizedCrop(cfg.local_size, scale=cfg.local_crops_scale,
                                          interpolation=bic, antialias=True)]
            + photometric + [blur(cfg.blur_prob_local), to_tensor] + morph_ops
        )

    def __call__(self, image):
        crops = [self.global_transfo1(image)]
        for _ in range(self.cfg.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.cfg.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


def build_transform(preset: AugPreset | str = TRAINING_PRESET, **overrides):
    """Build the training augmentation pipeline for a preset.

    Defaults to the locked training preset (``mild``). Returns a callable mapping
    a PIL image to a list of augmented crops (global + local views), matching the
    multi-crop training contract.
    """
    return MoleMultiCropAugmentation(resolve_config(preset, **overrides))


# --------------------------------------------------------------------------- #
# augview: visual inspection of augmentation strength (CPU-only, seconds).
# --------------------------------------------------------------------------- #
def _tensor_to_png_b64(t) -> str:
    from PIL import Image
    import numpy as np

    arr = (t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()).astype("uint8")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pil_to_png_b64(img, box: int) -> str:
    thumb = img.convert("RGB").copy()
    thumb.thumbnail((box, box))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pick_window_crops(image_path, img, window_size: int, n: int):
    """Pick ``n`` random foreground window crops from across the page.

    Each is a true training unit (a random ``window_size`` patch). Falls back to
    the whole image if it is smaller than ``window_size`` / has no foreground
    windows. Returns ``(crops, n_available_windows)``.
    """
    from mole.data.patches import sample_windows

    wins = sample_windows(image_path, window_size=window_size, overlap=0.5,
                          foreground_min=0.05)
    if not wins:
        return [img] * n, 0
    chosen = [random.choice(wins) for _ in range(n)]
    crops = [img.crop((w.x, w.y, w.x + w.size, w.y + w.size)) for w in chosen]
    return crops, len(wins)


def augview(folder: str, output: str, n_images: int = 5, n_views: int = 5,
            presets: list[AugPreset | str] | None = None, seed: int = 0,
            window_size: int = 512):
    """Write an HTML grid of ``n_images`` x ``n_views`` augmented crops per preset.

    Faithful to the training distribution: each view is a RANDOM ``window_size``
    patch-window sampled from across the page (the true training unit), then
    augmented — so you see both spatial (where on the page) and photometric
    variety, exactly as an epoch draws samples. The same window sequence is
    reused across presets so strengths stay comparable. The first column shows
    the whole page for context. Shows the global (model-size) view.

    Cheap, CPU-only, seconds to run.
    """
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    random.seed(seed)

    if presets is None:
        presets = [AugPreset.MILD, AugPreset.DEFAULT, AugPreset.AGGRESSIVE]

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No images found in {folder!r}")
    files = files[:n_images]
    images = [Image.open(f).convert("RGB") for f in files]
    # Pick the window sequence ONCE (same crops reused across presets for a fair
    # comparison); each view is a different random window from across the page.
    per_image = [_pick_window_crops(f, img, window_size, n_views)
                 for f, img in zip(files, images)]

    sections = []
    for preset in presets:
        cfg = resolve_config(preset)
        aug = MoleMultiCropAugmentation(cfg)
        rows = []
        for f, img, (crops, n_avail) in zip(files, images, per_image):
            cells = [f'<td class="orig"><img src="data:image/png;base64,{_pil_to_png_b64(img, cfg.model_size)}">'
                     f'<div class="cap">{f.name}<br>{img.size[0]}×{img.size[1]} · {n_avail} windows</div></td>']
            for crop in crops:
                view = aug.global_transfo1(crop)
                cells.append(f'<td><img src="data:image/png;base64,{_tensor_to_png_b64(view)}"></td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        headers = "<th>full page</th>" + "".join(f"<th>rand window {i+1}</th>" for i in range(n_views))
        key = preset.value if isinstance(preset, AugPreset) else str(preset)
        sections.append(
            f'<h2>preset: {key}</h2>'
            f'<div class="meta">window={window_size}px · global_scale={cfg.global_crops_scale} · rotation p={cfg.rotation_prob}'
            f' ±{cfg.rotation_degrees}° · grayscale p={cfg.grayscale_prob} · jitter p={cfg.color_jitter_prob}'
            f' · blur p={cfg.blur_prob_global1} · jpeg p={cfg.jpeg_prob} · erode/dilate p={cfg.erosion_dilation_prob}</div>'
            f'<table><tr>{headers}</tr>{"".join(rows)}</table>'
        )

    html = _AUGVIEW_HTML.replace("__BODY__", "\n".join(sections))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


_AUGVIEW_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mole augview</title>
<style>
 body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
 h1{font-weight:600} h2{margin-top:32px;border-bottom:1px solid #444;padding-bottom:4px}
 .meta{color:#9ab;font-size:13px;margin:6px 0 12px;font-family:ui-monospace,monospace}
 table{border-collapse:collapse} th{font-size:12px;color:#aaa;font-weight:500;padding:4px}
 td{padding:3px;text-align:center;vertical-align:top}
 td img{display:block;border-radius:3px;width:150px;height:150px;object-fit:contain;background:#222}
 td.orig img{outline:2px solid #6a9} .cap{font-size:10px;color:#888;max-width:150px;word-break:break-all}
</style></head>
<body><h1>mole augview — augmentation preview (global / model-size views)</h1>
__BODY__
</body></html>
"""
