"""Post-aggregation metric learning: the leave-one-archive-out contract.

The synthetic corpus hides writer identity in a low-variance subspace that a
high-variance nuisance direction dominates — the situation a supervised
projection is supposed to fix and whitening alone is not. Crucially the writer
subspace is SHARED across archives while the hands are disjoint, so a projection
learned on archives A/B must help archive C without ever seeing it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mole.supervised.docmetric import (
    PCAWhiten,
    archive_macro_map,
    fit_doc_metric,
    load_archive_vectors,
)


def _corpus(seed=0, archives=("a", "b", "c", "d", "e"), hands_per=30, docs_per=4,
            sig=8, noise=60):
    """Writer signal in `sig` shared dims, swamped by high-variance nuisance.

    The size is deliberately realistic (150 hands => 120 per training fold, 24
    for validation; the real pool has 175 and 140). An earlier 6-hand version left
    a 2-hand validation slice, so model selection was noise and the method looked
    broken — the corpus was, not the method. The margin over the whitening control
    grows with training hands (+0.05 at 60, +0.16 at 140), so sample size is part
    of the contract, not an incidental detail.
    """
    rng = np.random.default_rng(seed)
    dim = sig + noise
    X, hands, docs, archs = [], [], [], []
    for arch in archives:
        for h in range(hands_per):
            center = rng.standard_normal(sig)
            center /= np.linalg.norm(center)
            for d in range(docs_per):
                v = np.zeros(dim, np.float32)
                v[:sig] = center + 0.3 * rng.standard_normal(sig)
                v[sig:] = 4.0 * rng.standard_normal(noise)   # dominates the variance
                X.append(v)
                hands.append(f"{arch}/H{h}")
                docs.append(f"{arch}/d{h}_{d}")
                archs.append(arch)
    return (np.asarray(X, np.float32), np.asarray(hands, dtype=object),
            np.asarray(docs, dtype=object), np.asarray(archs, dtype=object))


def test_supervision_transfers_to_a_completely_unseen_archive():
    X, hands, docs, archives = _corpus()
    keep = archives == "c"

    raw, _ = archive_macro_map(X, hands, docs, keep)
    full = PCAWhiten().fit(X[archives != "c"])
    pca_macro, _ = archive_macro_map(full.transform(X), hands, docs, keep)

    # whiten_dim=None (full rank) on purpose: this corpus hides the writer signal
    # in the LOWEST-variance components, so truncating by variance would throw it
    # away — the failure mode PCAWhiten's docstring warns about.
    transform, report = fit_doc_metric(
        X, hands, docs, archives, holdout_archive="c",
        whiten_dim=None, out_dim=8, epochs=40, lr=1e-2, seed=0)
    sup, _ = archive_macro_map(transform(X), hands, docs, keep)

    # the held-out archive contributed nothing to whitening, loss or selection
    assert report["holdout_archive"] == "c"
    assert all(not h.startswith("c/") for h in report["train_hands"])
    assert all(not h.startswith("c/") for h in report["holdout_hands"])
    # ... and is still improved, over BOTH the raw space and whitening alone.
    # The second is the one that matters: whitening a VLAD descriptor is itself a
    # strong unsupervised trick, so supervision only earns credit beyond it.
    assert sup > raw + 0.05
    assert sup > pca_macro + 0.05


def test_truncating_by_variance_discards_low_variance_writer_signal():
    """Truncation loses signal — IN THIS REGIME ONLY (600 docs, 68 dims).

    Writer identity is low-variance, so with far more documents than dimensions a
    generous whiten_dim beats an aggressive one. ⚠ This does NOT generalise to the
    real data, where 38,400 dims over 3,392 documents makes near-full-rank
    whitening a noise amplifier (measured: -0.031 macro alone). The regime, not
    the rule, is what transfers — which is why fit_doc_metric defaults to 256.
    """
    X, hands, docs, archives = _corpus()
    keep = archives == "c"
    full = PCAWhiten().fit(X[archives != "c"])
    wide, _ = archive_macro_map(full.transform(X), hands, docs, keep)
    narrow, _ = archive_macro_map(full.truncated(8).transform(X), hands, docs, keep)
    assert wide > narrow
    # truncated() must equal a fresh fit at that dimensionality
    fresh = PCAWhiten(dim=8).fit(X[archives != "c"])
    assert np.allclose(full.truncated(8).transform(X), fresh.transform(X), atol=1e-5)


def test_whitening_is_fitted_without_the_held_out_archive():
    X, _, _, archives = _corpus()
    p = PCAWhiten(dim=6).fit(X[archives != "c"])
    q = PCAWhiten(dim=6).fit(X)
    # a transform fitted on everything differs from one that excluded c:
    # if these matched, the LOAO claim would be leaking through the whitening
    assert not np.allclose(p.mean_, q.mean_)


def test_cross_document_relevance_ignores_sibling_scans():
    """Two scans of one charter must not count as retrieving each other."""
    rng = np.random.default_rng(1)
    v = rng.standard_normal(8).astype(np.float32)
    X = np.stack([v, v + 0.001, rng.standard_normal(8).astype(np.float32)])
    hands = np.asarray(["a/H", "a/H", "a/G"], dtype=object)
    docs = np.asarray(["a/1", "a/1", "a/2"], dtype=object)   # first two are siblings
    macro, per_hand = archive_macro_map(X, hands, docs, np.ones(3, bool))
    # H's only "relevant" neighbour is its own sibling -> excluded -> no queries,
    # G is a singleton -> no queries either. Nothing scoreable, and no free credit.
    assert np.isnan(macro) or "a/H" not in per_hand


def test_load_rejects_mixed_codebooks(tmp_path):
    """Different dimensionalities cannot share a codebook — refuse, don't guess."""
    for name, dim in (("x", 8), ("y", 12)):
        ds = tmp_path / f"arch_{name}"
        ds.mkdir()
        (ds / "f.png").touch()
        (ds / "labels.csv").write_text("filename,hand_id\nf.png,H\n")
        np.save(tmp_path / f"{name}.npy", np.zeros((1, dim), np.float32))
        (tmp_path / f"{name}.mapping.json").write_text(
            json.dumps({"rows": [{"row": 0, "image": str(ds / "f.png")}]}))

    with pytest.raises(ValueError, match="codebook"):
        load_archive_vectors([tmp_path / "x.npy", tmp_path / "y.npy"])


def test_missing_sidecar_names_the_likely_cause(tmp_path):
    """A bare *.npy glob picks up mole embed's codebook artifacts — say so."""
    np.save(tmp_path / "a.codebook.npy", np.zeros((4, 8), np.float32))
    with pytest.raises(ValueError, match="codebook"):
        load_archive_vectors([tmp_path / "a.codebook.npy"])
