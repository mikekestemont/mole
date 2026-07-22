"""SGR reranking: it must help fragmented clusters, and must not cheat on siblings.

The second half is the one that matters on this corpus. 56 charters here are
several scans of one physical document; a page's nearest neighbour is very often
its own near-duplicate sibling, so a reranker that diffuses over them would
manufacture a gain out of exactly the scan shortcut ``--cross-doc-only`` exists
to suppress.
"""

from __future__ import annotations

import numpy as np

from mole.eval.rerank import apply_rerank, sgr_rerank


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def test_output_is_normalised_and_shape_preserving():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8)).astype(np.float32)
    before = X.copy()
    Y = sgr_rerank(X)
    assert Y.shape == X.shape
    np.testing.assert_allclose(np.linalg.norm(Y, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(X, before)                          # input untouched


def _fragmented(n_hands=40, frags=2, per_frag=3, d=64, sep=0.8, noise=0.9, seed=0):
    """Each hand splits into `frags` sub-clusters — the Flanders failure mode.

    FINCH on Flanders found 82 clusters at purity 0.891 but ARI 0.060: a hand's
    documents scattered across several individually *clean* fragments. That is
    what reranking is supposed to repair, so it is what the fixture models.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for h in range(n_hands):
        centre = rng.normal(size=d)
        for _ in range(frags):
            fc = centre + sep * rng.normal(size=d)
            for _ in range(per_frag):
                X.append(fc + noise * rng.normal(size=d))
                y.append(f"h{h}")
    return _l2(np.asarray(X, np.float32)), np.asarray(y, dtype=object)


def _macro_map(X, labels):
    from mole.eval.retrieval import _rank_metrics, _similarity

    sim = _similarity(X.astype(np.float64), "cosine")
    return _rank_metrics(sim, labels, ~np.eye(len(X), dtype=bool), (1,)).macro_map


def test_reranking_repairs_fragmented_clusters():
    """The claim is about RETRIEVAL, so measure retrieval, not descriptor geometry."""
    X, y = _fragmented()
    base = _macro_map(X, y)
    ranked = _macro_map(sgr_rerank(X, k=2, layers=1), y)
    assert base < 0.98, "fixture must leave headroom or the test proves nothing"
    assert ranked > base + 0.02, f"rerank should lift a fragmented set: {base:.4f} → {ranked:.4f}"


def test_gain_grows_with_difficulty():
    """Sanity on the mechanism: more scattered clusters ⇒ more for diffusion to fix."""
    gains = []
    for noise in (0.7, 1.1):
        X, y = _fragmented(noise=noise)
        gains.append(_macro_map(sgr_rerank(X, k=2, layers=1), y) - _macro_map(X, y))
    assert gains[1] > gains[0], f"expected a larger gain on harder data, got {gains}"


def test_siblings_are_excluded_from_the_graph():
    """A page must not be able to borrow from another scan of the same charter."""
    rng = np.random.default_rng(2)
    d = 12
    base = rng.normal(size=(1, d))
    # two near-identical sibling scans, plus unrelated pages
    X = np.vstack([base + 0.001 * rng.normal(size=(1, d)),
                   base + 0.001 * rng.normal(size=(1, d)),
                   rng.normal(size=(6, d))]).astype(np.float32)
    groups = np.array(["doc1", "doc1"] + [f"doc{i}" for i in range(2, 8)], dtype=object)

    free = sgr_rerank(X, k=1)
    guarded = sgr_rerank(X, k=1, groups=groups)

    # Unguarded, each sibling's nearest neighbour IS the other sibling, so the
    # pair collapses onto itself. Guarded, it must reach for something else.
    assert float(free[0] @ free[1]) > float(guarded[0] @ guarded[1])


def test_singleton_group_still_gets_neighbours():
    """Excluding siblings must not leave a page with an empty neighbourhood."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 6)).astype(np.float32)
    groups = np.array(["a", "a", "a", "b", "c"], dtype=object)
    Y = sgr_rerank(X, k=2, groups=groups)
    assert np.isfinite(Y).all()
    np.testing.assert_allclose(np.linalg.norm(Y, axis=1), 1.0, atol=1e-5)


def test_k_larger_than_available_neighbours():
    X = np.eye(3, dtype=np.float32)
    Y = sgr_rerank(X, k=10)
    assert Y.shape == (3, 3) and np.isfinite(Y).all()


def test_tiny_inputs_are_safe():
    assert sgr_rerank(np.zeros((0, 4), np.float32)).shape == (0, 4)
    assert sgr_rerank(np.ones((1, 4), np.float32)).shape == (1, 4)


def test_unknown_method_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown reranker"):
        apply_rerank(np.ones((3, 3), np.float32), "nope")


def test_eval_records_the_rerank_in_its_report(tmp_path):
    """Provenance: a reranked number must be self-identifying on disk.

    A gallery-dependent number that looks like a plain one is how two
    incomparable results end up in the same table.
    """
    import csv
    import json

    from mole.eval import evaluate

    ds = tmp_path / "arc"
    ds.mkdir()
    rng = np.random.default_rng(4)
    rows, vecs = [], []
    for h in range(4):
        centre = rng.normal(size=(8,))
        for j in range(3):
            name = f"h{h}_{j}.png"
            (ds / name).touch()
            rows.append({"filename": name, "hand_id": f"h{h}"})
            vecs.append(centre + rng.normal(scale=0.3, size=(8,)))
    with open(ds / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "hand_id"])
        w.writeheader()
        w.writerows(rows)

    npy = tmp_path / "e.npy"
    np.save(npy, np.asarray(vecs, dtype=np.float32))
    npy.with_suffix(".mapping.json").write_text(json.dumps({
        "model_id": "t@0", "pooling": "vlad",
        "rows": [{"row": i, "image": str(ds / r["filename"])} for i, r in enumerate(rows)],
    }))

    plain = evaluate(npy, ds, topk=(1,), out=tmp_path / "plain.json")
    ranked = evaluate(npy, ds, topk=(1,), rerank="sgr", out=tmp_path / "sgr.json")
    assert plain.rerank is None
    assert ranked.rerank == "sgr"
    assert json.loads((tmp_path / "sgr.json").read_text())["rerank"] == "sgr"
