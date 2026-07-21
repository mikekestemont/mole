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
    assert r.n_documents == 28

    # 1. the unlabeled document drawn from A is attributed to A
    top = next(a for a in r.attributions if a["document"] == "UNL_1.png")
    assert top["hand"] == "A"

    # 2. the two labels sharing one generator top the merge list
    m = r.merges[0]
    assert {m["hand_a"], m["hand_b"]} == {"TWIN1", "TWIN2"}
    assert m["closeness"] > -0.05               # as alike as each is to itself

    # 3. the label spanning two clouds tops the split list
    assert r.splits[0]["hand"] == "SPLITME"
    assert r.splits[0]["percentile"] >= 90      # sharper than 90% of random splits
    groups = (set(r.splits[0]["group_a"]), set(r.splits[0]["group_b"]))
    assert any(g == {"S10_1.png", "S11_1.png", "S12_1.png"} for g in groups)

    # 4. the mislabeled document tops the doubts list, pointing at B
    d = r.doubts[0]
    assert d["document"] == "MIS_1.png" and d["closer_hand"] == "B"

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
