"""Phase-1 gate tests for the supervised index, doc grouping, and pair masks."""

from __future__ import annotations

import numpy as np

from mole.eval.retrieval import load_hand_set
from mole.supervised.datasets import (
    SupervisedIndex,
    SupItem,
    load_labeled_pairs,
    pair_masks,
)


def _archive(root, name, rows, header="filename,hand_id"):
    """rows: list of csv line strings (without the header). Creates image files."""
    d = root / name
    d.mkdir(parents=True)
    for line in rows:
        fname = line.split(",")[0]
        (d / fname).write_bytes(b"")
    (d / "labels.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
    return d


# ------------------------------------------------------------ pair_masks (rule)
def test_pair_masks_positive_negative_ignore():
    # rows: 0,1 same hand different docs; 2 same hand SAME doc as 0; 3 other hand
    hands = ["A/x", "A/x", "A/x", "A/y"]
    docs = ["A/1", "A/2", "A/1", "A/9"]
    pos, neg = pair_masks(hands, docs)
    # positive = same hand, different doc
    assert pos[0, 1] and pos[1, 0]
    # same-doc same-hand (0 & 2) is in NEITHER mask (ignored)
    assert not pos[0, 2] and not neg[0, 2]
    # different hand -> negative, never positive
    assert neg[0, 3] and neg[3, 0]
    assert not pos[0, 3]
    # diagonal excluded from both
    assert not np.any(np.diag(pos)) and not np.any(np.diag(neg))


def test_pair_masks_namespacing_blocks_cross_archive_false_positive():
    # identical RAW hand "8" in two archives must NOT be a positive
    hands = ["flanders/8", "leroy/8"]
    docs = ["flanders/107", "leroy/1274"]
    pos, neg = pair_masks(hands, docs)
    assert not pos[0, 1] and not pos[1, 0]
    assert neg[0, 1]  # they are (correctly) a negative, not a positive


# ------------------------------------------------------- load_labeled_pairs
def test_namespacing_keeps_same_raw_hand_distinct(tmp_path):
    _archive(tmp_path, "arch1", ["a1.png,A", "a2.png,A"])
    _archive(tmp_path, "arch2", ["b1.png,A", "b2.png,A"])
    idx = load_labeled_pairs(tmp_path)
    assert set(idx.hands) == {"arch1/A", "arch2/A"}
    assert len(idx.by_hand["arch1/A"]) == 2


def test_min_confidence_demotes_to_unlabeled(tmp_path):
    _archive(tmp_path, "leroy-bin",
             ["a.png,5,0.90", "b.png,5,0.30", "c.png,,0.10"],
             header="filename,hand_id,confidence")
    idx = load_labeled_pairs(tmp_path, min_confidence=0.5)
    # b (0.30) demoted; c never labeled -> both unlabeled; only a survives
    assert [it.path.name for it in idx.items] == ["a.png"]
    assert len(idx.unlabeled) == 2


def test_leroy_gysseling_siblings_collapse_into_one_doc(tmp_path):
    # 117o & 118o share gysseling 170 -> one doc -> hand not retrievable on them alone
    _archive(tmp_path, "leroy-bin",
             ["117o.png,87,0.9,170", "118o.png,87,0.9,170",
              "956o.png,86,0.9,1146", "957o.png,86,0.9,2000"],
             header="filename,hand_id,match_score,gysseling_nr")
    idx = load_labeled_pairs(tmp_path)
    # hand 87 has 2 images but ONE document -> not retrievable
    assert len(idx.docs_by_hand["leroy-bin/87"]) == 1
    assert "leroy-bin/87" not in idx.retrievable_hands(min_docs=2)
    # hand 86 has two distinct gysseling docs -> retrievable
    assert "leroy-bin/86" in idx.retrievable_hands(min_docs=2)


def test_flanders_leading_number_groups_siblings(tmp_path):
    _archive(tmp_path, "flanders-set-bin",
             ["134_2_x.jpg,KA", "134_3_x.jpg,KA", "140_1_y.jpg,KA"])
    idx = load_labeled_pairs(tmp_path)
    docs = idx.docs_by_hand["flanders-set-bin/KA"]
    assert docs == {"flanders-set-bin/134", "flanders-set-bin/140"}


# --------------------------------------------------------------- split_hands
def _pooled(tmp_path):
    # two archives, several retrievable hands each (2 docs apiece)
    _archive(tmp_path, "arch1",
             [f"h{h}_{d}.png,H{h}" for h in range(4) for d in range(2)])
    _archive(tmp_path, "arch2",
             [f"g{h}_{d}.png,G{h}" for h in range(4) for d in range(2)])
    return load_labeled_pairs(tmp_path)


def test_split_is_deterministic_disjoint_and_stratified(tmp_path):
    idx = _pooled(tmp_path)
    train, hold = idx.split_hands(holdout_frac=0.25, seed=0)
    train2, hold2 = idx.split_hands(holdout_frac=0.25, seed=0)
    assert hold.hands == hold2.hands                    # deterministic
    assert not (set(train.hands) & set(hold.hands))     # disjoint
    assert set(train.hands) | set(hold.hands) == set(idx.hands)  # covering
    # stratified: each archive contributes ≥1 holdout hand (4 hands * 0.25 = 1)
    assert any(h.startswith("arch1/") for h in hold.hands)
    assert any(h.startswith("arch2/") for h in hold.hands)
    # a different seed generally picks a different holdout
    _, hold_s1 = idx.split_hands(holdout_frac=0.25, seed=1)
    assert hold.hands != hold_s1.hands


def test_write_holdout_split_roundtrips_to_the_eval_consumer(tmp_path):
    idx = _pooled(tmp_path)
    split = tmp_path / "sup_holdout_v1.json"
    _, hold = idx.write_holdout_split(split, holdout_frac=0.25, seed=0)
    loaded = load_hand_set(split)                       # the mole eval consumer
    assert loaded == set(hold.hands)
    # entries are namespaced 'archive/hand', matching how eval namespaces queries
    assert all("/" in h for h in loaded)


def test_stats_reports_multi_image_docs(tmp_path):
    _archive(tmp_path, "flanders-set-bin",
             ["134_2_x.jpg,KA", "134_3_x.jpg,KA", "140_1_y.jpg,KA", "150_1_z.jpg,KA"])
    idx = load_labeled_pairs(tmp_path)
    s = idx.stats()
    assert "flanders-set-bin/134" in s      # the collapsed sibling doc is shown
    assert "multi-image documents (siblings collapsed): 1" in s


def test_subset_reindexes_cleanly(tmp_path):
    idx = _pooled(tmp_path)
    sub = idx.subset({"arch1/H0", "arch1/H1"})
    assert set(sub.hands) == {"arch1/H0", "arch1/H1"}
    assert all(idx.items[i].hand in {"arch1/H0", "arch1/H1"}
               for idxs in sub.by_hand.values() for i in idxs)
    assert sub.unlabeled == []
