"""Foreground patch selection — Raven's exact rule, contrast, and polarity handling."""

from __future__ import annotations

import pytest
import torch

from mole.embed.extract import _foreground_mask, _window_foreground_mask

P = 16                      # patch size -> 256 px per patch
T_FG = 10 / 256             # PAPER: t_fg = 10 foreground pixels per patch token (3.9%)


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


def test_raven_patch_rule_uses_t_fg_of_10_pixels():
    # PAPER: t_fg = 10 pixels -> 9px drops, 10px keeps.
    mask = _foreground_mask(_window([0, 9, 10, 64]), P, T_FG, method="raven")
    assert mask[0].tolist() == [False, False, True, True]


def test_raven_rule_is_polarity_agnostic():
    # Same ink counts must give the same mask on white-on-black (raven/HWI native)
    # and black-on-white (Sauvola) — polarity is auto-detected as the minority tone.
    ink = [0, 9, 10, 64]
    wob = _foreground_mask(_window(ink, bright_ink=True), P, T_FG, method="raven")
    bow = _foreground_mask(_window(ink, bright_ink=False), P, T_FG, method="raven")
    assert wob[0].tolist() == bow[0].tolist() == [False, False, True, True]


def test_contrast_is_more_lenient_than_raven():
    # Documents the diagnosed HWI gap: contrast>0.05 admits near-empty patches that
    # Raven's t_fg=10px rule discards, diluting VLAD.
    ink = [0, 3, 64]                       # 3px = 1.2% ink, well under t_fg
    raven = _foreground_mask(_window(ink), P, T_FG, method="raven")[0].tolist()
    contrast = _foreground_mask(_window(ink), P, 0.05, method="contrast")[0].tolist()
    assert raven == [False, False, True]
    assert contrast == [False, True, True]   # keeps the 1.2% patch raven drops


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="raven"):
        _foreground_mask(_window([64]), P, 0.025, method="bogus")


# --------------------------------------------------------- window-level pre-filter

def _full_window(ink_frac, bright_ink=True, size=224):
    bg = 0.0 if bright_ink else 1.0
    img = torch.full((3, size, size), bg)
    flat = img.reshape(3, -1)
    flat[:, :round(ink_frac * size * size)] = 1.0 - bg
    return flat.reshape(3, size, size)


def test_window_filter_keeps_above_2_5_percent():
    # PAPER: "windows with MORE THAN 2.5% foreground pixels" -> strict >
    crops = [_full_window(f) for f in (0.0, 0.01, 0.025, 0.05, 0.30)]
    assert _window_foreground_mask(crops, 0.025, method="raven").tolist() == \
        [False, False, False, True, True]


def test_window_filter_is_polarity_agnostic():
    fracs = (0.0, 0.01, 0.025, 0.05, 0.30)
    wob = _window_foreground_mask([_full_window(f, True) for f in fracs], 0.025, method="raven")
    bow = _window_foreground_mask([_full_window(f, False) for f in fracs], 0.025, method="raven")
    assert wob.tolist() == bow.tolist()


def test_window_and_patch_thresholds_are_distinct():
    # The paper's two rules differ: windows >2.5%, patch tokens t_fg=10px (10/256=3.9%).
    # A window at 3% foreground clears the window filter, yet a 3%-ink patch (7-8px)
    # is still below the patch t_fg — the two thresholds are not interchangeable.
    assert _window_foreground_mask([_full_window(0.03)], 0.025, method="raven").tolist() == [True]
    assert _foreground_mask(_window([7, 8]), P, T_FG, method="raven")[0].tolist() == [False, False]
    assert _foreground_mask(_window([10, 64]), P, T_FG, method="raven")[0].tolist() == [True, True]


def test_unknown_window_method_raises():
    with pytest.raises(ValueError, match="raven"):
        _window_foreground_mask([_full_window(0.3)], 0.025, method="bogus")
