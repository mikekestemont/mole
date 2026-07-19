"""Tests for FeatureCache + HandBatchSampler (window-level P×D×W sampling)."""

from __future__ import annotations

import numpy as np

from mole.supervised.datasets import FeatureCache, HandBatchSampler, pair_masks


def _synthetic_cache(seed=0, windows_per_doc=4):
    """3 archives; several hands each with 2 docs; some unlabeled windows."""
    rng = np.random.default_rng(seed)
    hand, doc, arch, item = [], [], [], []
    layout = {"a": ["H0", "H1", "H2"], "b": ["G0", "G1", "G2"], "c": ["K0", "K1"]}
    for a, hands in layout.items():
        for h in hands:
            for d in range(2):                       # 2 docs per hand
                for _ in range(windows_per_doc):
                    hand.append(f"{a}/{h}")
                    doc.append(f"{a}/{h}d{d}")
                    arch.append(a)
                    item.append(f"{a}/{h}d{d}.png")
    for _ in range(7):                               # unlabeled windows
        hand.append(""); doc.append(""); arch.append("a"); item.append("unl.png")
    desc = rng.standard_normal((len(hand), 8)).astype(np.float32)
    return FeatureCache(desc, hand, doc, arch, item)


def test_batch_shape_and_labels_align():
    cache = _synthetic_cache()
    s = HandBatchSampler(cache, hands_per_batch=4, docs_per_hand=2,
                         windows_per_doc=3, same_archive_frac=0.5, seed=0)
    rows, hands, docs = next(iter(s))
    assert len(rows) == len(hands) == len(docs) == 4 * 2 * 3
    assert len(set(hands)) == 4                       # 4 distinct hands
    # each hand appears exactly docs_per_hand * windows_per_doc times
    _, counts = np.unique(hands, return_counts=True)
    assert set(counts.tolist()) == {2 * 3}
    # each hand spans exactly docs_per_hand distinct docs
    for h in set(hands):
        hd = {d for hh, d in zip(hands, docs) if hh == h}
        assert len(hd) == 2


def test_never_samples_unlabeled_windows():
    cache = _synthetic_cache()
    s = HandBatchSampler(cache, hands_per_batch=4, docs_per_hand=2,
                         windows_per_doc=3, seed=1)
    for rows, hands, _ in s:
        assert all(cache.window_hand[r] != "" for r in rows)
        assert all(h != "" for h in hands)


def test_every_anchor_has_a_positive():
    cache = _synthetic_cache()
    s = HandBatchSampler(cache, hands_per_batch=4, docs_per_hand=2,
                         windows_per_doc=3, seed=2)
    rows, hands, docs = next(iter(s))
    pos, _ = pair_masks(hands, docs)
    assert (pos.sum(axis=1) > 0).all()               # no anchor lacks a positive


def test_same_archive_fraction_enforced():
    cache = _synthetic_cache()
    s = HandBatchSampler(cache, hands_per_batch=4, docs_per_hand=2,
                         windows_per_doc=2, same_archive_frac=0.5, seed=3)
    for _, hands, _ in s:
        archs = [h.split("/", 1)[0] for h in dict.fromkeys(hands)]  # per distinct hand
        top = max(np.unique(archs, return_counts=True)[1])
        assert top >= 2                              # ≥ round(4*0.5) share an archive


def test_deterministic_with_seed():
    cache = _synthetic_cache()
    a = next(iter(HandBatchSampler(cache, hands_per_batch=4, seed=7)))
    b = next(iter(HandBatchSampler(cache, hands_per_batch=4, seed=7)))
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_windows_per_doc_uses_replacement_when_scarce():
    # only 2 windows per doc but we ask for 3 -> replacement fills the batch
    cache = _synthetic_cache(windows_per_doc=2)
    s = HandBatchSampler(cache, hands_per_batch=4, docs_per_hand=2,
                         windows_per_doc=3, seed=0)
    rows, hands, docs = next(iter(s))
    assert len(rows) == 4 * 2 * 3


def test_roundtrip_save_load(tmp_path):
    cache = _synthetic_cache()
    cache.meta = {"model_id": "t@0", "window_size": 224}
    cache.save(tmp_path / "cache")
    back = FeatureCache.load(tmp_path / "cache")
    assert back.n_windows == cache.n_windows and back.dim == cache.dim
    assert back.window_hand == cache.window_hand
    assert back.meta["model_id"] == "t@0"
    assert np.allclose(back.descriptors, cache.descriptors)
