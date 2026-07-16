"""Foreground patch selection — Raven's exact rule, contrast, and polarity handling."""

from __future__ import annotations

import pytest
import torch

from mole.embed.extract import _foreground_mask

P = 16                      # patch size -> 256 px per patch; 2.5% of 256 = 6.4 px


def _window(ink_pixels, bright_ink=True):
    """One window whose i-th patch contains ``ink_pixels[i]`` foreground pixels."""
    n = len(ink_pixels)
    bg = 0.0 if bright_ink else 1.0
    ink = 1.0 - bg
    img = torch.full((3, P, P * n), bg)
    for i, k in enumerate(ink_pixels):
        block = img[:, :, i * P:(i + 1) * P].reshape(3, -1)
        block[:, :k] = ink
        img[:, :, i * P:(i + 1) * P] = block.reshape(3, P, P)
    return [img]


def test_raven_rule_counts_foreground_pixels_at_2_5_percent():
    # 6/256 = 2.34% (below 2.5% -> drop); 7/256 = 2.73% (at/above -> keep)
    mask = _foreground_mask(_window([0, 6, 7, 64]), P, 0.025, method="raven")
    assert mask[0].tolist() == [False, False, True, True]


def test_raven_rule_is_polarity_agnostic():
    # Same ink fractions must give the same mask on white-on-black (raven/HWI native)
    # and black-on-white (Sauvola) — polarity is auto-detected as the minority tone.
    ink = [0, 6, 7, 64]
    wob = _foreground_mask(_window(ink, bright_ink=True), P, 0.025, method="raven")
    bow = _foreground_mask(_window(ink, bright_ink=False), P, 0.025, method="raven")
    assert wob[0].tolist() == bow[0].tolist() == [False, False, True, True]


def test_contrast_is_more_lenient_than_raven():
    # Documents the diagnosed HWI gap: contrast>0.05 admits near-empty patches that
    # Raven's 2.5% rule discards, diluting VLAD.
    ink = [0, 3, 64]                       # 3/256 = 1.2% ink
    raven = _foreground_mask(_window(ink), P, 0.025, method="raven")[0].tolist()
    contrast = _foreground_mask(_window(ink), P, 0.05, method="contrast")[0].tolist()
    assert raven == [False, False, True]
    assert contrast == [False, True, True]   # keeps the 1.2% patch raven drops


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="raven"):
        _foreground_mask(_window([64]), P, 0.025, method="bogus")
