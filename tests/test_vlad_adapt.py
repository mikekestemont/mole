"""Vocabulary adaptation (All About VLAD, CVPR 2013): keep the cells, move the centres."""

from __future__ import annotations

import numpy as np

from mole.embed import vlad as _vlad


def _two_blobs(rng, n=400, dim=6, shift=5.0):
    """Two well-separated Gaussian blobs, so nearest-centre assignment is unambiguous."""
    a = rng.standard_normal((n, dim)).astype(np.float32)
    b = rng.standard_normal((n, dim)).astype(np.float32) + shift
    return a, b


def test_adapt_moves_centre_to_cell_mean():
    rng = np.random.default_rng(0)
    a, b = _two_blobs(rng)
    x = np.vstack([a, b])
    # A codebook whose centres are OFFSET from the true blob means (the dataset-shift
    # the paper corrects): adaptation should pull each centre back to its cell's mean.
    codebook = np.stack([a.mean(0) + 1.3, b.mean(0) - 1.3]).astype(np.float32)
    adapted, counts = _vlad.adapt_codebook(codebook, x, min_assigned=10)

    assert counts.tolist() == [len(a), len(b)]           # clean split, blobs separated
    assert np.allclose(adapted[0], a.mean(0), atol=0.2)  # centre 0 -> blob A mean
    assert np.allclose(adapted[1], b.mean(0), atol=0.2)  # centre 1 -> blob B mean
    # and it genuinely moved toward the data
    assert np.linalg.norm(adapted[0] - a.mean(0)) < np.linalg.norm(codebook[0] - a.mean(0))


def test_assignment_uses_original_not_adapted_centres():
    # A degenerate case that only stays correct if assignment is done ONCE against the
    # original centres (a re-clustering would keep iterating and merge/steal points).
    rng = np.random.default_rng(1)
    a, b = _two_blobs(rng, shift=6.0)
    x = np.vstack([a, b])
    codebook = np.stack([a.mean(0), b.mean(0)]).astype(np.float32)
    adapted, counts = _vlad.adapt_codebook(codebook, x, min_assigned=1)
    # identity assignment preserved: counts equal the true blob sizes
    assert counts.tolist() == [len(a), len(b)]


def test_underpopulated_cell_keeps_frozen_centre():
    rng = np.random.default_rng(2)
    a, b = _two_blobs(rng)
    # give cell B only 3 descriptors, below the floor -> its centre must not move
    x = np.vstack([a, b[:3]])
    frozen_b = b.mean(0) - 2.0
    codebook = np.stack([a.mean(0) + 1.0, frozen_b]).astype(np.float32)
    adapted, counts = _vlad.adapt_codebook(codebook, x, min_assigned=10)

    assert counts[1] == 3
    assert np.array_equal(adapted[1], frozen_b)          # kept frozen (too few points)
    assert not np.array_equal(adapted[0], codebook[0])   # cell A had enough -> moved


def test_empty_target_returns_codebook_unchanged():
    codebook = np.arange(12, dtype=np.float32).reshape(3, 4)
    adapted, counts = _vlad.adapt_codebook(codebook, np.zeros((0, 4), np.float32))
    assert np.array_equal(adapted, codebook)
    assert counts.tolist() == [0, 0, 0]


def test_adapted_codebook_encodes_like_any_codebook():
    # The adapted codebook is an ordinary [K, dim] array: vlad_encode consumes it as-is,
    # and its cluster identities line up 1:1 with the frozen parent (same K, same order).
    rng = np.random.default_rng(3)
    a, b = _two_blobs(rng)
    x = np.vstack([a, b])
    codebook = np.stack([a.mean(0) + 1.0, b.mean(0) - 1.0]).astype(np.float32)
    adapted, _ = _vlad.adapt_codebook(codebook, x, min_assigned=10)
    v = _vlad.vlad_encode(x, adapted)
    assert v.shape == (adapted.shape[0] * adapted.shape[1],)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)  # final global L2
