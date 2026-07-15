"""Invert flag + its interaction with the foreground filter (mole.embed)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from mole.data.patches import load_rgb
from mole.embed.extract import _build_transform, _foreground_mask


def test_load_rgb_invert(tmp_path):
    p = tmp_path / "wob.png"
    # a mostly-black image (value 10) with one white pixel — white-on-black regime
    arr = np.full((32, 32, 3), 10, dtype=np.uint8)
    arr[0, 0] = 255
    Image.fromarray(arr).save(p)

    normal = np.asarray(load_rgb(p))
    inverted = np.asarray(load_rgb(p, invert=True))
    assert normal[5, 5, 0] == 10 and inverted[5, 5, 0] == 245   # 255 - 10
    assert normal[0, 0, 0] == 255 and inverted[0, 0, 0] == 0


def _patch_means(img, model_size=32, ps=16):
    crop = _build_transform(model_size)(img)          # [3, S, S] in [0,1]
    return torch.nn.functional.avg_pool2d(crop[0:1], ps).reshape(-1), crop


def test_contrast_foreground_drops_blank_parchment(tmp_path):
    """On parchment (background well below white), intensity keeps everything but
    contrast drops the smooth/blank patches and keeps the high-variance ink ones."""
    # 32x32, patch grid 2x2 (patch_size 16): one patch has "ink texture" (checker
    # -> high local std), the other three are flat mid-grey "parchment" (low std).
    arr = np.full((32, 32, 3), 150, dtype=np.uint8)     # flat parchment ~0.59
    arr[0:16:2, 0:16] = 20                              # striped ink in the top-left patch
    p = tmp_path / "parch.png"
    Image.fromarray(arr).save(p)
    crop = _build_transform(32)(load_rgb(p))

    # intensity: parchment (~0.59) is < 0.98 -> nothing dropped (Raven fails here)
    keep_int = _foreground_mask([crop], 16, 0.02, method="intensity")[0]
    assert bool(keep_int.all())

    # contrast: only the textured (ink) patch clears the std threshold
    keep_con = _foreground_mask([crop], 16, 0.06, method="contrast")[0]
    assert bool(keep_con[0]) and not bool(keep_con[1:].any())


def test_invert_fixes_foreground_on_white_on_black(tmp_path):
    """White-on-black: without invert the filter keeps background & drops ink;
    with invert it correctly keeps ink and drops the (now white) background."""
    # 32x32, patch grid 2x2. Top-left patch = ink (white strokes on black),
    # the other three patches = pure black background.
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[0:16, 0:16] = 200                              # bright "ink" block (white-on-black)
    p = tmp_path / "wob.png"
    Image.fromarray(arr).save(p)

    # As-is (white-on-black): background is black (~0) -> mean < 0.98 -> KEPT (wrong),
    # and the bright ink block trends toward being the thing dropped.
    _, crop_raw = _patch_means(load_rgb(p), 32, 16)
    keep_raw = _foreground_mask([crop_raw], 16, 0.02)[0]
    assert bool(keep_raw.all())                        # nothing dropped: filter is useless here

    # Inverted (black-on-white): background becomes white (~1) -> DROPPED; ink patch kept.
    _, crop_inv = _patch_means(load_rgb(p, invert=True), 32, 16)
    keep_inv = _foreground_mask([crop_inv], 16, 0.02)[0]
    assert bool(keep_inv[0]) and not bool(keep_inv[1:].any())  # only the ink patch survives
