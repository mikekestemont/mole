"""Each suggestion list is checked against a corpus built to contain the answer.

The embedding is synthesised so that the truth is known by construction: one
unlabeled document is drawn from hand A's generator, two labels are really one
scribe, one label is really two scribes, one labeled document is a mislabel, and
one pair is the same charter twice. Every list is then asked to put that planted
case first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mole.review import build_review


def _corpus(tmp_path: Path):
    """Write labels.csv + an embedding whose geometry encodes the planted truth."""
    rng = np.random.default_rng(0)
    dim = 32
    ds = tmp_path / "arch1"
    ds.mkdir()

    def cluster(center, n, spread=0.05):
        return center + spread * rng.standard_normal((n, dim))

    centers = {k: rng.standard_normal(dim) for k in ("A", "B", "TWIN", "FAR", "NEW")}
    vecs, names, hands = [], [], []

    def add(vec, name, hand):
        vecs.append(vec); names.append(name); hands.append(hand)

    # hand A: 4 documents, tight
    for i, v in enumerate(cluster(centers["A"], 4)):
        add(v, f"A{i}_1.png", "A")
    # hand B: 4 documents, tight and far from A
    for i, v in enumerate(cluster(centers["B"], 4)):
        add(v, f"B{i}_1.png", "B")
    # TWIN1 / TWIN2: two labels drawn from ONE generator -> should merge
    for i, v in enumerate(cluster(centers["TWIN"], 3)):
        add(v, f"T1{i}_1.png", "TWIN1")
    for i, v in enumerate(cluster(centers["TWIN"], 3)):
        add(v, f"T2{i}_1.png", "TWIN2")
    # SPLITME: one label over two well-separated groups -> should split
    for i, v in enumerate(cluster(centers["A"] * -1.0, 3)):
        add(v, f"S1{i}_1.png", "SPLITME")
    for i, v in enumerate(cluster(centers["FAR"], 3)):
        add(v, f"S2{i}_1.png", "SPLITME")
    # a mislabel: sits inside B's cloud but carries hand A's label
    add(centers["B"] + 0.05 * rng.standard_normal(dim), "MIS_1.png", "A")
    # an isolation outlier: labeled A but far from A and from every other hand
    add(centers["FAR"] + 0.05 * rng.standard_normal(dim), "ISOL_1.png", "A")
    # an unlabeled document drawn from A -> attribution to A
    add(centers["A"] + 0.05 * rng.standard_normal(dim), "UNL_1.png", "")
    # a group of unlabeled documents unlike anything known -> possible new hand
    for i, v in enumerate(cluster(centers["NEW"], 4)):
        add(v, f"N{i}_1.png", "")
    # the same charter twice under two names (different doc ids) -> duplicate
    dup = centers["B"] + 0.001 * rng.standard_normal(dim)
    add(dup, "D1_1.png", "")
    add(dup.copy(), "D2_1.png", "")

    for n in names:                              # load_labels validates against
        (ds / n).touch()                         # the images actually present
    rows = [{"row": i, "image": str(ds / n)} for i, n in enumerate(names)]
    X = np.asarray(vecs, dtype=np.float32)
    npy = tmp_path / "emb.npy"
    np.save(npy, X)
    (tmp_path / "emb.mapping.json").write_text(
        json.dumps({"model_id": "test@0", "rows": rows}))
    (ds / "labels.csv").write_text(
        "filename,hand_id\n"
        + "".join(f"{n},{h}\n" for n, h in zip(names, hands) if h))
    return npy


def test_every_list_surfaces_its_planted_case(tmp_path):
    r = build_review(_corpus(tmp_path), limit=20)

    assert r.n_hands == 5                       # A, B, TWIN1, TWIN2, SPLITME
    assert r.n_documents == 29

    # 1. the unlabeled document drawn from A is attributed to A.
    # Hands are namespaced by dataset folder, so two archives' hand "A" can never
    # collide into a false positive (same rule as mole.supervised.datasets).
    top = next(a for a in r.attributions if a["document"] == "UNL_1.png")
    assert top["hand"] == "arch1/A"

    # 2. the two labels sharing one generator top the merge list
    m = r.merges[0]
    assert {m["hand_a"], m["hand_b"]} == {"arch1/TWIN1", "arch1/TWIN2"}
    assert m["closeness"] > -0.05               # as alike as each is to itself

    # 3. the label spanning two clouds tops the split list
    assert r.splits[0]["hand"] == "arch1/SPLITME"
    assert r.splits[0]["percentile"] >= 90      # sharper than 90% of random splits
    groups = (set(r.splits[0]["group_a"]), set(r.splits[0]["group_b"]))
    assert any(g == {"S10_1.png", "S11_1.png", "S12_1.png"} for g in groups)

    # 4. the mislabeled document is a false-positive (closer to B than to A)
    d = next(o for o in r.outliers if o["document"] == "MIS_1.png")
    assert d["hand"] == "arch1/A" and d["closer_hand"] == "arch1/B"
    assert d["gap"] >= 0.05 and d["z"] >= 3.0

    # 5. the duplicated charter is found, and known siblings are not
    assert r.duplicates
    assert {r.duplicates[0]["document_a"], r.duplicates[0]["document_b"]} == \
        {"D1_1.png", "D2_1.png"}

    # 6. the unlabeled NEW cloud is proposed as a possible new hand
    assert any(set(c["documents"]) >= {"N0_1.png", "N1_1.png", "N2_1.png"}
               for c in r.new_hands)


def test_calibration_is_fitted_and_monotone(tmp_path):
    r = build_review(_corpus(tmp_path), limit=20)
    cal = r.calibration
    assert cal["fitted"] and cal["n"] >= 8
    p = cal["precision"]
    assert all(b >= a - 1e-9 for a, b in zip(p, p[1:]))    # isotonic => monotone
    assert all(0.0 <= v <= 1.0 for v in p)
    # attributions carry a probability, not a bare cosine
    assert all(a["calibrated_p"] is None or 0.0 <= a["calibrated_p"] <= 1.0
               for a in r.attributions)


def test_sibling_scans_are_not_evidence(tmp_path):
    """Two scans of one charter must not vouch for each other (flanders doc-id rule)."""
    rng = np.random.default_rng(1)
    dim = 16
    ds = tmp_path / "flanders-set-bin"           # doc id = leading number
    ds.mkdir()
    a = rng.standard_normal(dim)
    names = ["7_1_x.png", "7_2_x.png", "9_1_x.png"]
    vecs = [a, a + 0.001 * rng.standard_normal(dim), rng.standard_normal(dim)]
    for n in names:
        (ds / n).touch()
    rows = [{"row": i, "image": str(ds / n)} for i, n in enumerate(names)]
    np.save(tmp_path / "e.npy", np.asarray(vecs, dtype=np.float32))
    (tmp_path / "e.mapping.json").write_text(json.dumps({"rows": rows}))
    # only the FIRST scan is labeled; the second is its sibling
    (ds / "labels.csv").write_text("filename,hand_id\n7_1_x.png,H\n")

    r = build_review(tmp_path / "e.npy", limit=10)
    # 7_2 is a sibling of the only evidence for H, so it gets no attribution from it
    assert not any(a["document"] == "7_2_x.png" for a in r.attributions)
    # and the near-identical sibling pair is NOT reported as a duplicate
    assert not any({d["document_a"], d["document_b"]} == {"7_1_x.png", "7_2_x.png"}
                   for d in r.duplicates)


def test_build_review_does_not_write_labels(tmp_path):
    """labels.csv is the reference throughout — the engine must only read it."""
    npy = _corpus(tmp_path)
    lab = next(tmp_path.rglob("labels.csv"))
    before = lab.read_text()
    build_review(npy, limit=20)
    assert lab.read_text() == before


def test_lists_false_skips_suggestions_but_keeps_clusters(tmp_path):
    r = build_review(_corpus(tmp_path), lists=False)
    assert r.outliers == [] and r.attributions == []
    assert r.cluster_levels                          # viz still colours by FINCH


def test_outlier_cap_is_per_hand_not_global(tmp_path):
    """A dominant hand must not flood the queue and bury a small hand's outlier."""
    rng = np.random.default_rng(4)
    dim = 16
    ds = tmp_path / "arch1"
    ds.mkdir()
    vecs, names, hands = [], [], []

    def add(v, name, hand):
        vecs.append(v)
        names.append(name)
        hands.append(hand)

    big = rng.standard_normal(dim)
    small = rng.standard_normal(dim)
    for i in range(20):
        add(big + (0.35 if i < 12 else 0.05) * rng.standard_normal(dim),
            f"BIG{i}.png", "BIG")
    for i in range(4):
        add(small + 0.05 * rng.standard_normal(dim), f"S{i}.png", "SMALL")
    add(big + 0.05 * rng.standard_normal(dim), "SMALL_OUT.png", "SMALL")

    for n in names:
        (ds / n).touch()
    rows = [{"row": i, "image": str(ds / n)} for i, n in enumerate(names)]
    npy = tmp_path / "e.npy"
    np.save(npy, np.asarray(vecs, dtype=np.float32))
    (tmp_path / "e.mapping.json").write_text(json.dumps({"rows": rows}))
    (ds / "labels.csv").write_text(
        "filename,hand_id\n" + "".join(f"{n},{h}\n" for n, h in zip(names, hands)))

    r = build_review(npy, limit=40, per_hand_cap=8)
    by_hand = {}
    for o in r.outliers:
        by_hand.setdefault(o["hand"], []).append(o)
    assert all(len(v) <= 8 for v in by_hand.values())
    assert any(o["document"] == "SMALL_OUT.png" for o in r.outliers)


def test_false_negatives_skip_pages_that_would_be_outliers(tmp_path):
    """Keep on a false negative must not recreate a false positive."""
    r = build_review(_corpus(tmp_path), limit=40)
    assert any(a["document"] == "UNL_1.png" and a["hand"].endswith("/A")
               for a in r.attributions)
    # the unlabeled NEW cloud is unlike every known hand
    assert not any(a["document"].startswith("N") for a in r.attributions)
    for a in r.attributions:
        assert a["join_z"] < 1.0


def test_mild_closer_hand_is_not_a_misidentification(tmp_path):
    """A page still typical of its recorded hand must not enter stage 1 just
    because another hand is a hair closer. Recorded labels are the prior."""
    rng = np.random.default_rng(7)
    dim = 16
    ds = tmp_path / "arch1"
    ds.mkdir()
    vecs, names, hands = [], [], []
    a = rng.standard_normal(dim)
    b = rng.standard_normal(dim)
    for i in range(6):
        vecs.append(a + 0.04 * rng.standard_normal(dim))
        names.append(f"A{i}.png")
        hands.append("A")
    for i in range(6):
        vecs.append(b + 0.04 * rng.standard_normal(dim))
        names.append(f"B{i}.png")
        hands.append("B")
    # still inside A's cloud, nudged a little toward B
    vecs[0] = a + 0.03 * (b - a)
    for n in names:
        (ds / n).touch()
    rows = [{"row": i, "image": str(ds / n)} for i, n in enumerate(names)]
    npy = tmp_path / "e.npy"
    np.save(npy, np.asarray(vecs, dtype=np.float32))
    (npy.with_suffix(".mapping.json")).write_text(json.dumps({"rows": rows}))
    (ds / "labels.csv").write_text(
        "filename,hand_id\n" + "".join(f"{n},{h}\n" for n, h in zip(names, hands)))
    r = build_review(npy, limit=40)
    assert not any(o["document"] == "A0.png" for o in r.outliers)


def test_pick_partition_prefers_finch_ari_over_hdbscan():
    from mole.review.suggest import _pick_partition_for_new_hands

    levels = [
        {"level": "FINCH L0", "n_clusters": 12, "ari": 0.21, "labels": [0, 1, 2]},
        {"level": "FINCH L1", "n_clusters": 5, "ari": 0.77, "labels": [0, 0, 1]},
        {"level": "HDBSCAN min-size 2", "n_clusters": 4, "ari": 0.99, "labels": [9, 9, 9]},
    ]
    lab, meta = _pick_partition_for_new_hands(levels)
    assert meta["level"] == "FINCH L1" and meta["criterion"] == "ari"
    assert list(lab) == [0, 0, 1]


def test_new_hands_skip_clusters_that_overlap_labels():
    """A cluster that already contains a recorded hand is not a missed scribe."""
    from mole.review.suggest import _new_hands

    sim = np.eye(6, dtype=np.float32) * 0.2 + 0.8
    labels = np.array([0, 0, 0, 1, 1, 1])
    labeled = np.array([True, False, False, False, False, False])
    docs = np.asarray([f"d{i}" for i in range(6)], dtype=object)
    names = [f"p{i}.png" for i in range(6)]
    scores = np.full((6, 1), 0.1, dtype=np.float32)
    out = _new_hands(sim, labels, labeled, scores, docs, names, [0.5], 10)
    assert all(c["cluster"] == 1 for c in out)
    assert not any(c["cluster"] == 0 for c in out)
