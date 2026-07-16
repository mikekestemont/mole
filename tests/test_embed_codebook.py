"""External VLAD codebook reuse (fit-on-train / apply-on-test)."""

from __future__ import annotations

import numpy as np
import pytest

from mole.embed import vlad as _vlad
from mole.embed.extract import _assemble
from mole.embed.pooling import Pooling


def _pages(rng, n_pages=4, per_page=30, dim=8):
    return [rng.standard_normal((per_page, dim)).astype(np.float32) for _ in range(n_pages)]


def test_external_codebook_is_used_not_refit(tmp_path):
    rng = np.random.default_rng(0)
    train_pages = _pages(rng)
    dim = train_pages[0].shape[1]
    # "train" codebook (fit on a different set), saved as embed would save it
    train_cb = _vlad.fit_codebook(np.vstack(train_pages), n_clusters=5, seed=0)
    cb_path = tmp_path / "train.codebook.npy"
    np.save(cb_path, train_cb)

    test_pages = _pages(rng, n_pages=3)
    mat, used_cb = _assemble(Pooling.VLAD, [], test_pages, ["p0", "p1", "p2"], [],
                             vlad_clusters=999, seed=7, codebook_from=cb_path)
    # the loaded codebook is used verbatim (not a fresh 999-cluster fit)
    assert np.array_equal(used_cb, train_cb)
    assert mat.shape == (3, 5 * dim)
    # each row equals a manual encode against the train codebook
    for i, d in enumerate(test_pages):
        assert np.allclose(mat[i], _vlad.vlad_encode(d, train_cb), atol=1e-6)


def test_codebook_dim_mismatch_raises(tmp_path):
    cb_path = tmp_path / "bad.codebook.npy"
    np.save(cb_path, np.zeros((5, 16), dtype=np.float32))  # 16-dim vs 8-dim descriptors
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="expected .*8"):
        _assemble(Pooling.VLAD, [], _pages(rng), ["p0", "p1", "p2", "p3"], [],
                  vlad_clusters=5, seed=0, codebook_from=cb_path)


def test_streaming_vlad_matches_accumulate(tmp_path):
    # embed()'s external-codebook streaming path (encode per page, discard descriptors)
    # must be bit-identical to the accumulate-then-_assemble path — only memory differs.
    rng = np.random.default_rng(0)
    dim = 8
    pages = [rng.standard_normal((n, dim)).astype(np.float32) for n in (30, 12, 25)]
    codebook = _vlad.fit_codebook(np.vstack(pages), n_clusters=5, seed=0)
    cb_path = tmp_path / "cb.npy"; np.save(cb_path, codebook)
    imgs = ["a", "b", "c"]

    mat_accumulate, _ = _assemble(Pooling.VLAD, [], list(pages), list(imgs), [], 5, 0,
                                  intra_norm=False, codebook_from=cb_path)
    mat_stream = np.vstack([_vlad.vlad_encode(d, codebook, intra_norm=False) for d in pages])
    assert mat_accumulate.shape == mat_stream.shape
    assert np.array_equal(mat_accumulate, mat_stream)
