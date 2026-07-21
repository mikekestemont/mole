"""FINCH: the hierarchy is well-formed, recovers obvious structure, and scores it.

FINCH is used here to ask "how many hands are in the unlabeled pool?", so the
properties worth pinning are that it needs no K, that it actually separates
well-separated groups, and that agreement is measured over LABELED points only —
unlabeled documents are not negatives (an unlabeled charter may belong to a labeled
hand), so counting them as errors would bias every number we report.
"""

from __future__ import annotations

import numpy as np

from mole.cluster import cluster_agreement, finch


def _blobs(n_groups=5, per=12, dim=8, sep=12.0, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=sep, size=(n_groups, dim))
    x = np.repeat(centres, per, axis=0) + rng.normal(scale=0.25, size=(n_groups * per, dim))
    y = np.repeat(np.arange(n_groups), per)
    return x.astype(np.float32), y


def test_hierarchy_is_well_formed():
    x, _ = _blobs()
    res = finch(x, metric="euclidean")
    assert len(res.partitions) >= 1
    # strictly coarsening, one label per point at every level
    assert res.n_clusters == sorted(res.n_clusters, reverse=True)
    assert len(set(res.n_clusters)) == len(res.n_clusters)   # no stalled levels
    for p in res.partitions:
        assert p.shape == (len(x),)
        assert p.min() >= 0


def test_recovers_well_separated_groups():
    """Structure is recovered without being told K — but the exact K need not appear.

    FINCH forces every cluster to link to its nearest neighbour at each level, so a
    group that is already a single cluster gets absorbed into a neighbouring one and
    the true K can fall *between* levels (observed here: 12 -> 4 -> 1 for 5 groups).
    That is a property of the algorithm, not a failure, and it is why the CLI reports
    every level with its agreement rather than announcing one "best" K.

    What must hold: the finest partition never mixes groups, and some level aligns
    strongly with the truth.
    """
    x, y = _blobs(n_groups=5, per=12, sep=15.0)
    res = finch(x, metric="euclidean")
    true = [str(v) for v in y]
    scores = [cluster_agreement(true, p) for p in res.partitions]
    assert scores[0]["purity"] == 1.0, scores[0]        # fine level is pure
    assert max(s["nmi"] for s in scores) > 0.85, scores  # some level matches the truth


def test_agreement_ignores_unlabeled():
    """None entries must not affect the score, however many there are."""
    x, y = _blobs(n_groups=4, per=10, sep=15.0)
    res = finch(x, metric="euclidean")
    part = res.partitions[0]
    full = [str(v) for v in y]
    holed = [str(v) if i % 2 else None for i, v in enumerate(y)]   # half unlabeled
    a_full, a_holed = cluster_agreement(full, part), cluster_agreement(holed, part)
    assert a_full["n_labeled"] == len(y)
    assert a_holed["n_labeled"] == len(y) // 2
    # clustering is unchanged, so a clean partition still scores high on the subset
    assert a_holed["purity"] >= 0.9

    none_only = cluster_agreement([None] * len(y), part)
    assert none_only["n_labeled"] == 0 and none_only["purity"] is None


def test_tiny_and_degenerate_inputs():
    single = finch(np.zeros((1, 4), np.float32))
    assert len(single.partitions) == 1 and single.partitions[0].shape == (1,)
    dup = finch(np.ones((6, 3), np.float32), metric="euclidean")   # all identical
    assert dup.partitions[0].shape == (6,)
