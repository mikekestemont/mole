"""Tests for masked_supcon — the negative rule at the loss level."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mole.supervised.datasets import pair_masks
from mole.supervised.metric import masked_supcon


def test_ignored_pairs_enter_neither_numerator_nor_denominator():
    """A fully-ignored row must not affect any other anchor's loss."""
    rng = torch.Generator().manual_seed(0)
    z = torch.randn(4, 8, generator=rng)
    pos = torch.zeros(4, 4, dtype=torch.bool)
    neg = torch.zeros(4, 4, dtype=torch.bool)
    pos[0, 1] = pos[1, 0] = True                 # 0,1 are a positive pair
    neg[0, 2] = neg[2, 0] = True                 # 2 is a confirmed negative to 0
    neg[1, 2] = neg[2, 1] = True
    # row 3 is ignored by everyone (all-False columns) and has no positives.
    loss_a = masked_supcon(z, pos, neg).item()
    z2 = z.clone()
    z2[3] = torch.randn(8, generator=rng) * 5    # move the ignored row anywhere
    loss_b = masked_supcon(z2, pos, neg).item()
    assert abs(loss_a - loss_b) < 1e-6


def test_confirmed_negative_changes_loss_but_only_through_the_denominator():
    """Moving a confirmed negative closer to an anchor raises the loss."""
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])   # 0,1 same; 2 apart
    pos = torch.zeros(3, 3, dtype=torch.bool); pos[0, 1] = pos[1, 0] = True
    neg = torch.zeros(3, 3, dtype=torch.bool)
    neg[0, 2] = neg[2, 0] = neg[1, 2] = neg[2, 1] = True
    far = masked_supcon(z, pos, neg, temperature=0.1).item()
    z_close = z.clone(); z_close[2] = torch.tensor([0.9, 0.44])  # negative pulled in
    near = masked_supcon(z_close, pos, neg, temperature=0.1).item()
    assert near > far


def test_perfect_separation_is_near_zero_loss():
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    pos, neg = pair_masks(["A", "A", "B", "B"], ["d0", "d1", "d2", "d3"])
    loss = masked_supcon(z, torch.from_numpy(pos), torch.from_numpy(neg),
                         temperature=0.05).item()
    assert loss < 0.01


def test_no_positive_batch_is_zero_and_differentiable():
    z = torch.randn(3, 4, requires_grad=True)
    pos = torch.zeros(3, 3, dtype=torch.bool)
    neg = torch.ones(3, 3, dtype=torch.bool) & ~torch.eye(3, dtype=torch.bool)
    loss = masked_supcon(z, pos, neg)
    assert loss.item() == 0.0
    loss.backward()                              # graph intact, gradient is zero
    assert torch.allclose(z.grad, torch.zeros_like(z.grad))


def test_integrates_with_pair_masks_on_a_realistic_batch():
    rng = torch.Generator().manual_seed(1)
    hands = ["a/H0", "a/H0", "a/H1", "a/H1", "b/G0", "b/G0"]
    docs = ["a/0", "a/1", "a/2", "a/3", "b/4", "b/5"]
    pos, neg = pair_masks(hands, docs)
    z = torch.randn(6, 16, generator=rng)
    loss = masked_supcon(z, torch.from_numpy(pos), torch.from_numpy(neg))
    assert torch.isfinite(loss) and loss.item() > 0
