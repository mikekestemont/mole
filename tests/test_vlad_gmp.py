"""Generalized max pooling inside VLAD (`--vlad-pooling gmp`).

Tim Raven's VLAD class defaults to GMP (gamma=1000) — unpublished, his released
inference script uses sum — so the option exists to test that variant. It stays
opt-out: the default is the paper's sum pooling."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from mole.embed.vlad import GMP_GAMMA_DEFAULT, gmp_pool, vlad_encode


def _ridge_reference(r, gamma):
    # Raven: sklearn Ridge(alpha=gamma, fit_intercept=False).fit(residuals, ones).coef_
    n, d = r.shape
    return np.linalg.solve(r.T @ r + gamma * np.eye(d), r.T @ np.ones(n))


def test_gmp_is_the_ridge_solution():
    rng = np.random.default_rng(0)
    for n in (2, 40, 900):                         # under- and over-determined
        r = rng.normal(size=(n, 32)).astype(np.float32) * 3
        np.testing.assert_allclose(gmp_pool(r, 7.0), _ridge_reference(r, 7.0),
                                   rtol=1e-4, atol=1e-6)


def test_gmp_matches_sklearn_ridge_when_available():
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    r = rng.normal(size=(300, 48)).astype(np.float32)
    ref = sk.Ridge(alpha=1000.0, fit_intercept=False).fit(r, np.ones(300)).coef_
    np.testing.assert_allclose(gmp_pool(r, 1000.0), ref, rtol=1e-4, atol=1e-7)


def test_large_gamma_tends_to_sum_pooling():
    # xi -> R^T 1 / gamma: the plain sum up to a scale the global L2 removes.
    rng = np.random.default_rng(2)
    r = rng.normal(size=(100, 16))
    s = r.sum(0)
    x = gmp_pool(r, 1e9)
    assert x @ s / (np.linalg.norm(x) * np.linalg.norm(s)) > 0.99999


def test_gmp_suppresses_burstiness():
    # 100 copies of one descriptor own a sum-pooled block; GMP gives it one vote.
    rng = np.random.default_rng(3)
    base = rng.normal(size=(20, 16))
    burst = np.vstack([base, np.repeat(base[:1], 100, 0)])
    cos = lambda v: v @ base[0] / (np.linalg.norm(v) * np.linalg.norm(base[0]))
    assert cos(burst.sum(0)) > 0.99
    assert cos(gmp_pool(burst, 1.0)) < 0.5


def test_gmp_empty_cluster_is_zero():
    assert np.all(gmp_pool(np.zeros((0, 8)), 1.0) == 0)


def test_vlad_encode_pooling_option():
    rng = np.random.default_rng(4)
    cb = rng.normal(size=(5, 8)).astype(np.float32)
    x = rng.normal(size=(60, 8)).astype(np.float32)
    v_sum = vlad_encode(x, cb, intra_norm=False)                 # default = sum
    v_sum2 = vlad_encode(x, cb, intra_norm=False, pooling="sum")
    v_gmp = vlad_encode(x, cb, intra_norm=False, pooling="gmp", gmp_gamma=1.0)
    np.testing.assert_array_equal(v_sum, v_sum2)
    assert v_gmp.shape == v_sum.shape and not np.allclose(v_gmp, v_sum)
    assert np.isclose(np.linalg.norm(v_gmp), 1.0, atol=1e-5)  # still globally L2-normed
    # gmp at a huge gamma == sum after the scale-free normalisations
    v_big = vlad_encode(x, cb, intra_norm=False, pooling="gmp", gmp_gamma=1e12)
    np.testing.assert_allclose(v_big, v_sum, atol=1e-4)
    with pytest.raises(ValueError, match="pooling"):
        vlad_encode(x, cb, pooling="max")


def test_default_gamma_is_ravens():
    assert GMP_GAMMA_DEFAULT == 1000.0
    assert inspect.signature(vlad_encode).parameters["pooling"].default == "sum"


# ------------------------------------------------------------ pipeline plumbing
torch = pytest.importorskip("torch")

from mole.embed.extract import _warn_on_version_mismatch, _write_output, embed  # noqa: E402
from mole.embed.pooling import Pooling  # noqa: E402


def test_cli_and_api_default_to_sum():
    from mole.cli.main import embed as cli_embed
    assert inspect.signature(cli_embed).parameters["vlad_pooling"].default.default == "sum"
    assert inspect.signature(embed).parameters["vlad_pooling"].default == "sum"


def test_sidecar_records_pooling_and_gamma(tmp_path):
    cb = np.zeros((3, 4), np.float32)
    meta = {"model_id": "m@abcd1234+step0", "embed_dim": 4}
    _write_output(tmp_path / "g.npy", np.zeros((2, 12), np.float32),
                  [{"row": 0, "image": "a"}, {"row": 1, "image": "b"}], meta,
                  Pooling.VLAD, False, cb, 3, 0, vlad_intra_norm=False,
                  vlad_pooling="gmp", gmp_gamma=250.0)
    side = json.loads((tmp_path / "g.mapping.json").read_text())
    assert side["vlad_pooling"] == "gmp" and side["vlad_gmp_gamma"] == 250.0

    _write_output(tmp_path / "s.npy", np.zeros((2, 12), np.float32),
                  [{"row": 0, "image": "a"}, {"row": 1, "image": "b"}], meta,
                  Pooling.VLAD, False, cb, 3, 0, vlad_intra_norm=False)
    side = json.loads((tmp_path / "s.mapping.json").read_text())
    assert side["vlad_pooling"] == "sum" and "vlad_gmp_gamma" not in side


def test_guard_warns_on_mixed_pooling(tmp_path, capsys):
    (tmp_path / "old.mapping.json").write_text(json.dumps(
        {"model_id": "m@abcd1234+step0", "pooling": "vlad", "vlad_pooling": "sum"}))
    _warn_on_version_mismatch(tmp_path, "m@abcd1234+step0", pooling=Pooling.VLAD,
                              vlad_intra_norm=False, vlad_pooling="gmp")
    assert "vlad_pooling" in capsys.readouterr().out


def test_guard_treats_old_sidecars_as_sum(tmp_path, capsys):
    # Sidecars written before the option existed are sum-pooled: no warning for sum.
    (tmp_path / "old.mapping.json").write_text(json.dumps(
        {"model_id": "m@abcd1234+step0", "pooling": "vlad", "vlad_intra_norm": False}))
    _warn_on_version_mismatch(tmp_path, "m@abcd1234+step0", pooling=Pooling.VLAD,
                              vlad_intra_norm=False, vlad_pooling="sum")
    assert capsys.readouterr().out == ""
