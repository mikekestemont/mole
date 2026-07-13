"""Zone-aware patch-window dataset for self-supervised training.

The training unit is a ``window_size`` patch lifted from a page, restricted to the
prep text zone when a ``zones.json`` is present. Window coordinates are precomputed
from image *sizes* (no pixels loaded at init — fixes the original's
load-everything-to-RAM), and each ``__getitem__`` loads one image, crops one
window, and returns the multi-crop augmented views.

Masks are NOT produced here: AttMask needs the teacher's attention, so masks are
generated in the training loop. The dataset only exposes ``get_pred_ratio`` /
``set_epoch`` (the masking-ratio schedule), matching the original contract.

Returns ``(crops, 0)`` per sample (dummy label; labels never touch self-sup). The
default DataLoader collate turns a batch into ``[B×globals…, B×locals…]`` because
every sample shares the same crop structure (one preset).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from torch.utils.data import Dataset

from mole.data.augment import build_transform, resolve_config
from mole.data.datasets import IMAGE_EXTENSIONS
from mole.data.patches import Window, load_rgb, window_coords
from mole.data.zones import find_zones, load_zones


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


class PatchWindowDataset(Dataset):
    """Sliding-window patches over a dataset folder (flat or subfolders-as-datasets)."""

    def __init__(self, root: str | Path, window_size: int = 512, model_size: int = 224,
                 overlap: float = 0.5, use_zones: bool = True, preset="mild",
                 aug_overrides: dict | None = None, pred_ratio=0.3, pred_ratio_var=0.0,
                 pred_start_epoch: int = 0):
        root = Path(root)
        overrides = {**(aug_overrides or {}), "model_size": model_size}
        # Resolved AugConfig is the source of truth for global/local crop counts
        # (read by the training loop / loss); the transform is built from it.
        self.aug_config = resolve_config(preset, **overrides)
        self.transform = build_transform(preset, **overrides)
        self.window_size = window_size

        # Build the (image_path, window) index from sizes only — no pixel loads.
        folders = [root] + [p for p in sorted(root.iterdir()) if p.is_dir()] if root.is_dir() else []
        self.index: list[tuple[Path, Window]] = []
        for folder in folders:
            images = _list_images(folder)
            if not images:
                continue
            manifest = load_zones(find_zones(folder)) if use_zones and find_zones(folder) else None
            for img in images:
                bbox = manifest.bbox_for(img.name) if manifest else None
                size = None
                if manifest and img.name in manifest.images:
                    size = manifest.images[img.name].size
                if not size:
                    from PIL import Image, ImageFile
                    ImageFile.LOAD_TRUNCATED_IMAGES = True
                    size = Image.open(img).size
                for win in window_coords(size[0], size[1], window_size, overlap, bbox):
                    self.index.append((img, win))

        if not self.index:
            raise FileNotFoundError(f"No images/windows found under {root!r}")

        # Masking-ratio schedule (ported from ImageFolderMask).
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = (pred_ratio_var[0] if isinstance(pred_ratio_var, list)
                               and len(pred_ratio_var) == 1 else pred_ratio_var)
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.pred_start_epoch = pred_start_epoch
        self.epoch = 0

    @property
    def n_images(self) -> int:
        return len({p for p, _ in self.index})

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def get_pred_ratio(self):
        if self.epoch < self.pred_start_epoch:
            return 0
        if isinstance(self.pred_ratio, list):
            ratios = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                ratios.append(random.uniform(prm - prv, prm + prv) if prv > 0 else prm)
            return random.choice(ratios)
        assert self.pred_ratio >= self.pred_ratio_var
        if self.pred_ratio_var > 0:
            return random.uniform(self.pred_ratio - self.pred_ratio_var,
                                  self.pred_ratio + self.pred_ratio_var)
        return self.pred_ratio

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        path, win = self.index[idx]
        img = load_rgb(path)
        crop = img.crop((win.x, win.y, win.x + win.size, win.y + win.size))
        return self.transform(crop), 0
