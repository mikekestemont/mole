"""Phase-2B tests: build_feature_cache, the Tier-1 head trainer, train_metric."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from mole.supervised.datasets import (
    FeatureCache,
    build_feature_cache,
    load_labeled_pairs,
    window_descriptors,
)
from mole.supervised.metric import build_head, holdout_macro_map, train_head, train_metric


# --------------------------------------------------------- window_descriptors
def test_window_descriptors_mean_and_blank_skip():
    patches = np.array([
        [[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]],   # window 0: keep patches 0,1 -> mean [2,2]
        [[5.0, 5.0], [7.0, 7.0], [0.0, 0.0]],   # window 1: no foreground -> None
    ], dtype=np.float32)
    keep = np.array([[True, True, False], [False, False, False]])
    out = window_descriptors(patches, keep)
    assert np.allclose(out[0], [2.0, 2.0])
    assert out[1] is None


# ------------------------------------------------------------ build_feature_cache
def _tiny_checkpoint(tmp_path):
    from mole.selfsup.vit import build_vit
    model = build_vit("vit_tiny", patch_size=16, num_class_tokens=1)
    ck = tmp_path / "ck.pth"
    torch.save({"state_dict": model.state_dict()}, ck)
    return ck


def test_build_feature_cache_end_to_end(tmp_path):
    ck = _tiny_checkpoint(tmp_path)
    ds = tmp_path / "arch1"
    ds.mkdir()
    names = ["a_1.png", "a_2.png", "b_1.png", "b_2.png"]
    hands = ["A", "A", "B", "B"]
    rng = np.random.default_rng(0)
    for n in names:
        Image.fromarray((rng.random((260, 260, 3)) * 255).astype("uint8")).save(ds / n)
    (ds / "labels.csv").write_text(
        "filename,hand_id\n" + "".join(f"{n},{h}\n" for n, h in zip(names, hands)))

    index = load_labeled_pairs(ds)
    cache = build_feature_cache(ck, index, tmp_path / "cache", window_size=224,
                                overlap=0.0, invert=False, batch_size=8, progress=False)
    assert cache.dim == 192                       # vit_tiny embed_dim
    assert cache.n_windows >= 4                   # ≥1 window per image
    assert set(cache.window_hand) == {"arch1/A", "arch1/B"}
    assert cache.meta["fg_method"] == "contrast"
    # roundtrips through disk
    assert FeatureCache.load(tmp_path / "cache").n_windows == cache.n_windows


# -------------------------------------------------- the head improves geometry
def _signal_noise_cache(seed=0, n_hands=6, docs=3, wins=6, sig=6, noise=18):
    """Hand identity lives in a low-variance signal subspace swamped by noise.

    Raw cosine is noise-dominated (poor retrieval); a linear head can learn to
    project onto the signal dims — and, because that projection is hand-agnostic,
    it must help HELD-OUT hands too (the whole Tier-1 claim)."""
    rng = np.random.default_rng(seed)
    dim = sig + noise
    hand_sig = {f"a/H{h}": rng.standard_normal(sig) for h in range(n_hands)}
    for k in hand_sig:
        hand_sig[k] /= np.linalg.norm(hand_sig[k])
    H, D, A, I, X = [], [], [], [], []
    for h in range(n_hands):
        hid = f"a/H{h}"
        for d in range(docs):
            for _ in range(wins):
                v = np.zeros(dim, np.float32)
                v[:sig] = 0.6 * hand_sig[hid] + 0.1 * rng.standard_normal(sig)
                v[sig:] = 1.0 * rng.standard_normal(noise)          # swamping noise
                X.append(v); H.append(hid); D.append(f"a/H{h}d{d}")
                A.append("a"); I.append(f"{hid}d{d}.png")
    return FeatureCache(np.asarray(X, np.float32), H, D, A, I,
                        meta={"model_id": "tiny@0"})


def test_train_head_improves_held_out_hands():
    cache = _signal_noise_cache(seed=0)
    holdout = {"a/H4", "a/H5"}                    # 2 unseen hands (need distractors)
    torch.manual_seed(0)
    baseline = holdout_macro_map(cache.filter(holdout), torch.nn.Identity(), "cpu")
    head, report = train_head(
        cache, holdout_hands=holdout, out_dim=8, kind="linear", epochs=40, lr=1e-2,
        sampler_cfg={"hands_per_batch": 4, "docs_per_hand": 2, "windows_per_doc": 4,
                     "same_archive_frac": 0.0, "batches_per_epoch": 20},
        seed=0, progress=False)
    trained = report["best_holdout_macro"]
    assert trained > baseline + 0.05             # the head demonstrably helps unseen hands
    assert trained > 0.6
    # model-selection kept the best epoch, not merely the last
    assert trained >= max(h["holdout_macro"] for h in report["history"]) - 1e-9
    # no held-out hand leaked into training
    assert set(report["holdout_hands"]) == holdout
    assert holdout.isdisjoint(report["train_hands"])


def test_cache_labeled_only_skips_the_unlabeled_pool(tmp_path):
    ck = _tiny_checkpoint(tmp_path)
    ds = tmp_path / "arch1"
    ds.mkdir()
    rng = np.random.default_rng(0)
    for n in ["a_1.png", "a_2.png", "unlabeled_1.png"]:
        Image.fromarray((rng.random((260, 260, 3)) * 255).astype("uint8")).save(ds / n)
    (ds / "labels.csv").write_text("filename,hand_id\na_1.png,A\na_2.png,A\n")

    index = load_labeled_pairs(ds)
    assert len(index.unlabeled) == 1
    full = build_feature_cache(ck, index, tmp_path / "c_full", window_size=224,
                               overlap=0.0, invert=False, batch_size=8, progress=False)
    lean = build_feature_cache(ck, index, tmp_path / "c_lean", window_size=224,
                               overlap=0.0, invert=False, batch_size=8, progress=False,
                               include_unlabeled=False)
    assert "" in full.window_hand                 # the unlabeled window is there
    assert "" not in lean.window_hand             # ... and skipped when asked
    assert lean.n_windows < full.n_windows
    assert lean.meta["include_unlabeled"] is False


def test_shipped_config_sampler_keys_reach_the_sampler():
    """configs/sup_head.yaml `sup.sampler` is splatted into HandBatchSampler.

    A typo there would only raise on the server, AFTER the one GPU cache pass —
    so check the contract here instead.
    """
    import inspect
    from pathlib import Path

    from mole.config import load_config
    from mole.supervised.datasets import HandBatchSampler
    from mole.supervised.metric import _SUP_DEFAULTS

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "sup_head.yaml"
    sup = load_config(cfg_path)["sup"]
    assert set(sup) <= set(_SUP_DEFAULTS), "unknown sup.* key would be silently ignored"
    accepted = set(inspect.signature(HandBatchSampler.__init__).parameters) - {"self", "cache"}
    assert set(sup["sampler"]) <= accepted, "sampler key HandBatchSampler would reject"


def test_build_head_shapes():
    lin = build_head("linear", 384, 128)
    assert lin(torch.zeros(5, 384)).shape == (5, 128)
    mlp = build_head("mlp", 384, 128)
    assert mlp(torch.zeros(5, 384)).shape == (5, 128)
    with pytest.raises(ValueError):
        build_head("bogus", 384, 128)


# ------------------------------------------------------------- train_metric wiring
def test_train_metric_writes_head_report_split(tmp_path):
    ck = _tiny_checkpoint(tmp_path)
    ds = tmp_path / "arch1"
    ds.mkdir()
    rng = np.random.default_rng(1)
    names, hands = [], []
    for h in range(4):                            # 4 hands x 2 docs (single image each)
        for d in range(2):
            n = f"h{h}_d{d}.png"
            Image.fromarray((rng.random((250, 250, 3)) * 255).astype("uint8")).save(ds / n)
            names.append(n); hands.append(f"H{h}")
    (ds / "labels.csv").write_text(
        "filename,hand_id\n" + "".join(f"{n},{h}\n" for n, h in zip(names, hands)))

    cfg = tmp_path / "sup.yaml"
    cfg.write_text(
        "sup:\n  tier: head\n  head: linear\n  out_dim: 8\n  epochs: 3\n"
        "  holdout_frac: 0.25\n  seed: 0\n"
        "  sampler: {hands_per_batch: 2, docs_per_hand: 2, windows_per_doc: 2, same_archive_frac: 0.0}\n")

    out = tmp_path / "run"
    head_path = train_metric(cfg, ck, ds, out)
    assert head_path.is_file()
    assert (out / "report.json").is_file()
    assert (out / "split.json").is_file()
    blob = torch.load(head_path, weights_only=False)
    assert blob["in_dim"] == 192 and blob["out_dim"] == 8 and blob["kind"] == "linear"
    assert blob["base_model_id"] == "vit_tiny@" + blob["base_model_id"].split("@", 1)[1]
