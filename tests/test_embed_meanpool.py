"""Codebook-free mean / mean+std page poolings."""

from __future__ import annotations

import numpy as np

from mole.embed.extract import _fixed_vector_dim, _mean_pool
from mole.embed.pooling import Pooling


def test_enum_has_meanstd():
    assert Pooling("meanstd") is Pooling.MEANSTD
    assert Pooling("mean") is Pooling.MEAN


def test_mean_is_l2_normed_and_dimensioned():
    desc = np.random.default_rng(0).standard_normal((300, 384)).astype(np.float32)
    v = _mean_pool(desc, with_std=False, embed_dim=384)
    assert v.shape == (384,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_meanstd_blocks_are_separately_normed():
    desc = np.random.default_rng(1).standard_normal((300, 384)).astype(np.float32)
    v = _mean_pool(desc, with_std=True, embed_dim=384)
    assert v.shape == (768,)
    assert np.isclose(np.linalg.norm(v[:384]), 1.0, atol=1e-5)   # mean block
    assert np.isclose(np.linalg.norm(v[384:]), 1.0, atol=1e-5)   # std block
    # mean-only is exactly the first block of meanstd
    assert np.allclose(v[:384], _mean_pool(desc, with_std=False, embed_dim=384))


def test_std_block_actually_carries_spread():
    # Two token sets with an IDENTICAL mean but different per-dimension spread must be
    # identical under plain mean yet differ under meanstd — the whole point of the +std.
    # (The spread must differ in *shape* across dims: per-block L2 cancels a global
    # scale, so only the relative spread pattern survives — which is what we want.)
    rng = np.random.default_rng(2)
    d, n = 16, 2000
    mean_vec = rng.standard_normal(d).astype(np.float32)
    spread_a = np.array([3.0] * 8 + [0.3] * 8, np.float32)
    spread_b = np.array([0.3] * 8 + [3.0] * 8, np.float32)   # same magnitudes, flipped
    a = rng.standard_normal((n, d)).astype(np.float32) * spread_a
    b = rng.standard_normal((n, d)).astype(np.float32) * spread_b
    a = a - a.mean(0) + mean_vec                              # force EXACT common mean
    b = b - b.mean(0) + mean_vec
    assert np.allclose(_mean_pool(a, False, d), _mean_pool(b, False, d), atol=1e-5)
    assert not np.allclose(_mean_pool(a, True, d), _mean_pool(b, True, d), atol=1e-2)


def test_empty_page_returns_zero_vector_of_right_width():
    assert _mean_pool(np.zeros((0, 384), np.float32), False, 384).shape == (384,)
    assert _mean_pool(np.zeros((0, 384), np.float32), True, 384).shape == (768,)
    assert not _mean_pool(np.zeros((0, 384), np.float32), True, 384).any()


def test_fixed_vector_dim():
    assert _fixed_vector_dim(Pooling.MEAN, 384, 1) == 384
    assert _fixed_vector_dim(Pooling.MEANSTD, 384, 1) == 768
    assert _fixed_vector_dim(Pooling.CLS, 384, 2) == 768   # num_class_tokens * dim
