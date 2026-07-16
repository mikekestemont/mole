"""Polarity-invariant contrast foreground: patch mask, window fractions, dataset filter."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from mole.data.patches import (Window, patch_contrast_mask,
                                window_foreground_fractions)
from mole.selfsup.dataset import PatchWindowDataset


def _textured_and_blank_crop(patch=16):
    """A [1,1,2*patch,2*patch] crop: left half high-contrast stripes, right half flat."""
    a = np.full((2 * patch, 2 * patch), 0.5, np.float32)     # flat mid-grey = blank
    a[:, :patch] = np.tile([0.0, 1.0], (2 * patch, patch // 2))  # vertical stripes = inked
    return torch.from_numpy(a)[None, None]


def test_patch_contrast_mask_keeps_inked_drops_blank():
    x = _textured_and_blank_crop(patch=16)                   # 2x2 patch grid
    mask = patch_contrast_mask(x, patch_size=16)             # -> [1, 4]
    grid = mask.reshape(2, 2)
    assert grid[:, 0].all()          # left patches (stripes) kept
    assert not grid[:, 1].any()      # right patches (flat) dropped


def test_patch_contrast_mask_is_polarity_invariant():
    x = _textured_and_blank_crop(patch=16)
    assert torch.equal(patch_contrast_mask(x, 16), patch_contrast_mask(1.0 - x, 16))


def _page_with_ink_block():
    """160x160 page: a textured 40px block near the top-left, rest blank white."""
    a = np.full((160, 160), 255, np.uint8)
    a[8:48, 8:48] = np.tile([0, 255], (40, 20))              # stripes = ink
    return Image.fromarray(a, "L")


def test_window_fractions_separate_ink_from_blank_and_ignore_polarity():
    img = _page_with_ink_block()
    wins = [Window(0, 0, 64), Window(96, 96, 64)]            # ink corner vs blank corner
    fr = window_foreground_fractions(img, wins)
    assert fr[0] > 0.1 and fr[1] < 0.01
    inv = window_foreground_fractions(Image.eval(img, lambda p: 255 - p), wins)
    assert np.allclose(fr, inv)                              # polarity-invariant


def test_dataset_foreground_min_drops_blank_windows(tmp_path):
    for i in range(2):
        _page_with_ink_block().convert("RGB").save(tmp_path / f"p{i}.png")
    common = dict(window_size=32, overlap=0.0, use_zones=False)
    full = PatchWindowDataset(tmp_path, foreground_min=0.0, **common)
    filt = PatchWindowDataset(tmp_path, foreground_min=0.10, **common)
    assert 0 < len(filt) < len(full)                        # blanks dropped, ink kept
