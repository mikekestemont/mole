"""Tests for the retrieval benchmark (mole.eval.retrieval)."""

from __future__ import annotations

import json

import numpy as np

from mole.eval.retrieval import (
    _rank_metrics,
    _similarity,
    evaluate,
)


def test_rank_metrics_hand_computed():
    """mAP/Top-1/macro against a fully hand-worked 4-doc, 2-hand example."""
    labels = np.array(["A", "A", "B", "B"], dtype=object)
    # sim rows chosen so each query's ranking of the *others* is known:
    #  q0(A): 1(A),2,3   -> relevant first        -> AP 1.000, top1 hit
    #  q1(A): 0(A),2,3   -> relevant first        -> AP 1.000, top1 hit
    #  q2(B): 0,1,3(B)   -> relevant at rank 3    -> AP 0.333, top1 miss
    #  q3(B): 2(B),0,1   -> relevant first        -> AP 1.000, top1 hit
    sim = np.array([
        [0.00, 0.90, 0.50, 0.10],
        [0.90, 0.00, 0.40, 0.20],
        [0.80, 0.30, 0.00, 0.20],
        [0.20, 0.10, 0.95, 0.00],
    ])
    off_diag = ~np.eye(4, dtype=bool)
    s = _rank_metrics(sim, labels, off_diag, (1, 5))
    assert s is not None
    assert s.n_queries == 4
    assert s.mean_ap == np.float64(np.mean([1.0, 1.0, 1 / 3, 1.0])).item()
    assert abs(s.mean_ap - 0.83333) < 1e-4
    assert abs(s.top1 - 0.75) < 1e-9
    # macro = mean(hand A mean-AP=1.0, hand B mean-AP=(1/3+1)/2=0.6667)
    assert abs(s.macro_map - ((1.0 + (1 / 3 + 1.0) / 2) / 2)) < 1e-9
    assert abs(s.topk[5] - 1.0) < 1e-9  # every query has a relevant within top-(n-1)


def test_per_hand_ap_is_serialized():
    """Per-hand AP is exposed on RetrievalScores and macro == mean of it."""
    labels = np.array(["A", "A", "B", "B"], dtype=object)
    sim = np.array([
        [0.00, 0.90, 0.50, 0.10],
        [0.90, 0.00, 0.40, 0.20],
        [0.80, 0.30, 0.00, 0.20],
        [0.20, 0.10, 0.95, 0.00],
    ])
    s = _rank_metrics(sim, labels, ~np.eye(4, dtype=bool), (1,))
    assert set(s.per_hand) == {"A", "B"}
    assert abs(s.per_hand["A"]["ap"] - 1.0) < 1e-9
    assert abs(s.per_hand["B"]["ap"] - (1 / 3 + 1.0) / 2) < 1e-9
    assert s.per_hand["A"]["n_queries"] == 2 and s.per_hand["B"]["n_queries"] == 2
    # macro is exactly the mean of the per-hand AP values
    assert abs(s.macro_map - np.mean([s.per_hand[h]["ap"] for h in ("A", "B")])) < 1e-12


def test_min_confidence_demotes_low_rows_to_unlabeled(tmp_path):
    """A label below the floor drops out entirely — not counted as a negative."""
    ds = tmp_path / "wi"
    ds.mkdir(parents=True)
    names = ["a1.png", "a2.png", "b1.png", "b2.png"]
    for n in names:
        (ds / n).write_bytes(b"")
    # a2 is a low-confidence auto-match; the rest are trusted (confidence 0.9).
    (ds / "labels.csv").write_text(
        "filename,hand_id,confidence\n"
        "a1.png,A,0.90\na2.png,A,0.30\nb1.png,B,0.90\nb2.png,B,0.95\n")
    mat = np.asarray([[1, 0], [1, 0.01], [0, 1], [0.01, 1]], dtype=np.float32)
    npy = tmp_path / "emb.npy"
    np.save(npy, mat)
    (tmp_path / "emb.mapping.json").write_text(json.dumps({
        "model_id": "t@0",
        "rows": [{"row": i, "image": f"{ds}/{n}"} for i, n in enumerate(names)],
    }))

    # Without a floor: hand A has two docs, so A is a valid query.
    full = evaluate(npy, ds, topk=(1,))
    assert full.n_labeled == 4
    assert "A" in full.overall.per_hand
    # With a floor of 0.5: a2 is demoted -> A is now a singleton -> skipped as a
    # query, and a2 is not in the gallery either (only B remains retrievable).
    floored = evaluate(npy, ds, topk=(1,), min_confidence=0.5)
    assert floored.n_labeled == 3
    assert floored.min_confidence == 0.5
    assert "A" not in floored.overall.per_hand


def test_cross_doc_only_excludes_sibling_scans(tmp_path):
    """A sibling scan (same charter) must not count as a cross-document hit."""
    ds = tmp_path / "flanders-set-bin"
    ds.mkdir(parents=True)
    # Hand A: two scans of ONE charter (134) + one scan of another (140).
    # Hand B: two distinct charters (200, 201).
    names = ["134_2_x.jpg", "134_3_x.jpg", "140_1_y.jpg",
             "200_1_z.jpg", "201_1_w.jpg"]
    hands = ["A", "A", "A", "B", "B"]
    for n in names:
        (ds / n).write_bytes(b"")
    (ds / "labels.csv").write_text(
        "filename,hand_id\n" + "".join(f"{n},{h}\n" for n, h in zip(names, hands)))
    # Geometry (angles on the unit circle): the 134 siblings sit at ~0-5°, the
    # second A charter (140) sits far away at 90°, and the two B docs sit at
    # ~40-42° — *between* the 134 pair and 140. So with ordinary relevance a 134
    # query trivially retrieves its sibling at rank 1 (the scan shortcut); under
    # cross-doc-only the sibling is gone and the B docs outrank the distant 140,
    # so hand A's AP and Top-1 both drop.
    mat = np.asarray([
        [1.0000, 0.0000],   # 134_2  (0°)
        [0.9962, 0.0872],   # 134_3  (5°, sibling of 134_2)
        [0.0000, 1.0000],   # 140_1  (90°, second A charter, far away)
        [0.7660, 0.6430],   # 200_1  (40°, hand B)
        [0.7430, 0.6690],   # 201_1  (42°, hand B)
    ], dtype=np.float32)
    npy = tmp_path / "emb.npy"
    np.save(npy, mat)
    (tmp_path / "emb.mapping.json").write_text(json.dumps({
        "model_id": "t@0",
        "rows": [{"row": i, "image": f"{ds}/{n}"} for i, n in enumerate(names)],
    }))

    # Standard relevance: the 134 siblings retrieve each other -> hand A looks great.
    std = evaluate(npy, ds, topk=(1,))
    # Cross-doc-only: 134 siblings are excluded, so 134_2/134_3 must reach 140
    # (which is far away) -> hand A's AP collapses; the flag demonstrably bites.
    xdoc = evaluate(npy, ds, topk=(1,), cross_doc_only=True)
    assert xdoc.cross_doc_only is True
    assert xdoc.overall.per_hand["A"]["ap"] < std.overall.per_hand["A"]["ap"]
    # 134 queries no longer count their sibling as a top-1 hit
    assert xdoc.overall.top1 < std.overall.top1


def test_cross_doc_only_noop_when_all_docs_unique(tmp_path):
    """With one image per charter, cross-doc-only equals the standard metric."""
    ds = tmp_path / "brackley-set"
    ds.mkdir(parents=True)
    names = ["Brackley_D4.jpg", "Brackley_D5.jpg", "Brackley_D6.jpg", "Brackley_D7.jpg"]
    hands = ["A", "A", "B", "B"]
    for n in names:
        (ds / n).write_bytes(b"")
    (ds / "labels.csv").write_text(
        "filename,hand_id\n" + "".join(f"{n},{h}\n" for n, h in zip(names, hands)))
    mat = np.asarray([[1, 0], [1, 0.02], [0, 1], [0.02, 1]], dtype=np.float32)
    npy = tmp_path / "emb.npy"
    np.save(npy, mat)
    (tmp_path / "emb.mapping.json").write_text(json.dumps({
        "model_id": "t@0",
        "rows": [{"row": i, "image": f"{ds}/{n}"} for i, n in enumerate(names)],
    }))
    std = evaluate(npy, ds, topk=(1,))
    xdoc = evaluate(npy, ds, topk=(1,), cross_doc_only=True)
    assert abs(std.overall.mean_ap - xdoc.overall.mean_ap) < 1e-12
    assert std.overall.n_queries == xdoc.overall.n_queries


def test_holdout_hands_restricts_queries_not_gallery(tmp_path):
    """Held-out eval scores only held-out-hand queries, against the full gallery."""
    ds = tmp_path / "wi"
    ds.mkdir(parents=True)
    names = ["a1.png", "a2.png", "b1.png", "b2.png", "c1.png", "c2.png"]
    hands = ["A", "A", "B", "B", "C", "C"]
    for n in names:
        (ds / n).write_bytes(b"")
    (ds / "labels.csv").write_text(
        "filename,hand_id\n" + "".join(f"{n},{h}\n" for n, h in zip(names, hands)))
    mat = np.asarray([[1, 0, 0], [1, 0.02, 0], [0, 1, 0],
                      [0, 1, 0.02], [0, 0, 1], [0.02, 0, 1]], dtype=np.float32)
    npy = tmp_path / "emb.npy"
    np.save(npy, mat)
    (tmp_path / "emb.mapping.json").write_text(json.dumps({
        "model_id": "t@0",
        "rows": [{"row": i, "image": f"{ds}/{n}"} for i, n in enumerate(names)],
    }))

    from mole.eval.retrieval import load_hand_set
    split = tmp_path / "holdout.json"
    split.write_text(json.dumps({"holdout_hands": ["C"]}))
    held = load_hand_set(split)

    full = evaluate(npy, ds, topk=(1,))
    r = evaluate(npy, ds, topk=(1,), holdout_hands=held)
    assert full.overall.n_queries == 6      # A,B,C all query
    assert r.overall.n_queries == 2         # only the two C docs query
    assert set(r.overall.per_hand) == {"C"}
    assert r.n_holdout_hands == 1
    # gallery is still full: a namespaced entry matches the same way
    split.write_text(json.dumps({"holdout_hands": ["wi/C"]}))
    r2 = evaluate(npy, ds, topk=(1,), holdout_hands=load_hand_set(split))
    assert r2.overall.n_queries == 2


def test_singleton_hand_query_is_skipped():
    """A hand with only one labeled doc yields no relevant gallery item -> skipped."""
    labels = np.array(["A", "A", "Z"], dtype=object)  # Z is a singleton
    sim = np.array([[0, 0.9, 0.8], [0.9, 0, 0.7], [0.8, 0.7, 0]])
    s = _rank_metrics(sim, labels, ~np.eye(3, dtype=bool), (1,))
    assert s.n_queries == 2  # only the two A-docs are valid queries


def test_cosine_euclidean_agree_when_normalized():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 5))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # For unit vectors, cosine and (negative) euclidean induce identical rankings.
    cos = _similarity(X, "cosine")
    euc = _similarity(X, "euclidean")
    for i in range(6):
        assert np.array_equal(np.argsort(-cos[i]), np.argsort(-euc[i]))


def _write_dataset(root, names, hands, vectors):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"")  # dummy image file (suffix is all that matters)
    lines = ["filename,hand_id"]
    lines += [f"{n},{h}" for n, h in zip(names, hands) if h]
    (root / "labels.csv").write_text("\n".join(lines) + "\n")
    return np.asarray(vectors, dtype=np.float32)


def test_evaluate_end_to_end_perfect(tmp_path):
    """Perfectly separable embeddings -> mAP == Top-1 == 1.0, sidecar written."""
    ds = tmp_path / "wi"
    names = ["a1.png", "a2.png", "b1.png", "b2.png"]
    hands = ["A", "A", "B", "B"]
    vecs = [[1, 0], [1, 0.01], [0, 1], [0.01, 1]]  # same-hand vectors near-identical
    mat = _write_dataset(ds, names, hands, vecs)

    npy = tmp_path / "emb.npy"
    np.save(npy, mat)
    (tmp_path / "emb.mapping.json").write_text(json.dumps({
        "model_id": "test@abc+step0",
        "rows": [{"row": i, "image": f"{ds}/{n}"} for i, n in enumerate(names)],
    }))

    r = evaluate(npy, ds, topk=(1, 2))
    assert r.n_labeled == 4 and r.n_hands == 2
    assert abs(r.overall.mean_ap - 1.0) < 1e-9
    assert abs(r.overall.top1 - 1.0) < 1e-9
    assert (tmp_path / "emb.eval.json").is_file()
    saved = json.loads((tmp_path / "emb.eval.json").read_text())
    assert saved["overall"]["topk"]["1"] == 1.0  # json keys are strings


def test_evaluate_cross_dataset_breakdown(tmp_path):
    """Two datasets sharing a hand -> both within- and cross-dataset are scored.

    Note the two scans share the basename ``a*.png`` deliberately: attribution
    must key on the dataset folder, not just the basename.
    """
    root = tmp_path / "root"
    # hand A appears in BOTH scans (twice in scan1 so within-dataset is defined)
    _write_dataset(root / "scan1", ["a1.png", "a2.png", "b.png"],
                   ["A", "A", "B"], [[1, 0], [1, 0.02], [0, 1]])
    _write_dataset(root / "scan2", ["a3.png", "c.png"],
                   ["A", "C"], [[0.99, 0.01], [-1, 0]])

    mat = np.array([[1, 0], [1, 0.02], [0, 1], [0.99, 0.01], [-1, 0]], dtype=np.float32)
    npy = tmp_path / "emb.npy"
    np.save(npy, mat)
    rows = [
        {"row": 0, "image": f"{root}/scan1/a1.png"},
        {"row": 1, "image": f"{root}/scan1/a2.png"},
        {"row": 2, "image": f"{root}/scan1/b.png"},
        {"row": 3, "image": f"{root}/scan2/a3.png"},
        {"row": 4, "image": f"{root}/scan2/c.png"},
    ]
    (tmp_path / "emb.mapping.json").write_text(json.dumps({"model_id": "t@0", "rows": rows}))

    r = evaluate(npy, root, topk=(1,))
    assert set(r.datasets) == {"scan1", "scan2"}
    assert r.within_dataset is not None and r.cross_dataset is not None
    # within-dataset: only the two scan1 A-docs have a same-scan relative
    assert r.within_dataset.n_queries == 2
    # cross-dataset: the three A-docs each have a same-hand doc in the other scan
    assert r.cross_dataset.n_queries == 3
    assert abs(r.cross_dataset.top1 - 1.0) < 1e-9  # A-scans are nearest across sets
