"""PCA-whitening: transductive fit, and fit-on-train / apply-on-test (--whiten-from)."""

from __future__ import annotations

import numpy as np

from mole.embed.extract import _apply_pca_whiten, _fit_pca_whiten


def test_fit_reduces_dim_and_l2_normalises():
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((50, 400)).astype(np.float32)
    white, transform = _fit_pca_whiten(mat, dim=16)
    assert white.shape == (50, 16)
    assert np.allclose(np.linalg.norm(white, axis=1), 1.0, atol=1e-4)
    assert transform["proj"].shape == (400, 16) and transform["mean"].shape == (1, 400)


def test_apply_on_train_reproduces_the_fit():
    # Saving the train-fit transform and applying it back to train must reproduce
    # the transductive fit exactly — the invariant --whiten-from relies on.
    rng = np.random.default_rng(1)
    train = rng.standard_normal((60, 200)).astype(np.float32)
    white, transform = _fit_pca_whiten(train, dim=24)
    applied = _apply_pca_whiten(train, transform)
    assert np.allclose(applied, white, atol=1e-5)


def test_apply_on_test_is_independent_of_test_distribution():
    # The test embedding must depend only on the train-fit transform, not on which
    # other test rows are present (unlike a transductive re-fit).
    rng = np.random.default_rng(2)
    train = rng.standard_normal((80, 120)).astype(np.float32)
    _, transform = _fit_pca_whiten(train, dim=10)
    test = rng.standard_normal((30, 120)).astype(np.float32)
    row0_alone = _apply_pca_whiten(test[:1], transform)
    row0_in_batch = _apply_pca_whiten(test, transform)[:1]
    assert np.allclose(row0_alone, row0_in_batch, atol=1e-5)


def test_dim_mismatch_raises():
    rng = np.random.default_rng(3)
    _, transform = _fit_pca_whiten(rng.standard_normal((40, 128)).astype(np.float32), dim=8)
    try:
        _apply_pca_whiten(rng.standard_normal((5, 64)).astype(np.float32), transform)
        assert False, "expected ValueError on dim mismatch"
    except ValueError as e:
        assert "128-dim input" in str(e)
