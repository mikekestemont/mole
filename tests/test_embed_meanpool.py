"""Codebook-free mean / mean+std page poolings."""

from __future__ import annotations

import numpy as np

from mole.embed.extract import _cov_pool, _fixed_vector_dim, _mean_pool
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
    assert _fixed_vector_dim(Pooling.COV, 384, 1) == 384 * 385 // 2   # upper triangle


def test_cov_is_l2_normed_and_correctly_dimensioned():
    desc = np.random.default_rng(0).standard_normal((1000, 32)).astype(np.float32)
    v = _cov_pool(desc, 32)
    assert v.shape == (32 * 33 // 2,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_cov_sees_correlation_that_meanstd_cannot():
    # Two token clouds with matched per-dimension mean AND variance, differing ONLY in
    # cross-dimensional correlation: meanstd is identical, cov differs. This is the
    # whole reason cov exists (meanstd was a no-op on real data).
    rng = np.random.default_rng(1)
    n = 8000
    z = rng.standard_normal((n, 2)).astype(np.float32)
    a = z.copy()
    b = (z @ np.linalg.cholesky(np.array([[1.0, 0.9], [0.9, 1.0]], np.float32)).T).astype(np.float32)
    for m in (a, b):
        m -= m.mean(0); m /= m.std(0)      # standardise each dim: mean 0, var 1
    a = a + np.array([2.0, 1.0], np.float32)   # common non-zero mean -> stable, identical
    b = b + np.array([2.0, 1.0], np.float32)   #   mean block; only correlation now differs
    assert np.allclose(_mean_pool(a, True, 2), _mean_pool(b, True, 2), atol=1e-2)   # meanstd blind
    assert not np.allclose(_cov_pool(a, 2), _cov_pool(b, 2), atol=1e-2)             # cov sees it


def test_cov_flattening_preserves_frobenius_norm():
    # off-diagonal sqrt(2) weighting => the flattened upper triangle has the same norm
    # as the full symmetric matrix (Frobenius), so the descriptor loses nothing.
    d = 24
    desc = np.random.default_rng(2).standard_normal((500, d)).astype(np.float32)
    g = (desc.T @ desc) / len(desc)
    g = np.sign(g) * np.sqrt(np.abs(g))
    iu = np.triu_indices(d)
    u = g[iu].astype(np.float64).copy()
    u[iu[0] != iu[1]] *= np.sqrt(2.0)
    assert np.isclose(np.linalg.norm(u), np.linalg.norm(g), atol=1e-3)


def test_cov_empty_page_zero_vector():
    z = _cov_pool(np.zeros((0, 32), np.float32), 32)
    assert z.shape == (32 * 33 // 2,) and not z.any()
