"""The fully-differentiable VLAD forward: a metric loss on page doc-vectors must
backprop into BOTH the ViT backbone and the NetVLAD centroids (the whole point of
joint training — Tier-2's frozen-backbone version couldn't move the features)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mole.supervised.jointvlad import batch_docvectors, page_docvector
from mole.supervised.metric import masked_supcon
from mole.supervised.netvlad import NetVLAD


def _tiny_setup(k=6, seed=0):
    from mole.selfsup.vit import build_vit
    model = build_vit("vit_tiny", patch_size=16, return_all_tokens=True, num_class_tokens=1)
    model.train()
    dim = int(model.embed_dim)
    rng = np.random.default_rng(seed)
    codebook = rng.standard_normal((k, dim)).astype(np.float32)
    netvlad = NetVLAD.from_codebook(codebook, alpha=50.0, learn="both")
    netvlad.train()
    return model, netvlad, dim


def _page(rng, n_windows, s=64):
    # n_windows crops of [C, S, S] in [0,1], as the deterministic resize would produce.
    return [torch.from_numpy(rng.random((3, s, s)).astype("float32")) for _ in range(n_windows)]


def test_page_docvector_shape_and_grad():
    model, netvlad, dim = _tiny_setup()
    rng = np.random.default_rng(1)
    v = page_docvector(model, netvlad, _page(rng, 3), num_class_tokens=1, patch_size=16,
                       fg_threshold=0.0, fg_method="contrast", embed_dim=dim,
                       max_tokens=0, max_windows=0, rng=rng, device="cpu")
    assert v.shape == (netvlad.centroids.shape[0] * dim,)
    assert v.requires_grad                                 # attached to the graph


def test_loss_backprops_into_backbone_and_centroids():
    model, netvlad, dim = _tiny_setup()
    rng = np.random.default_rng(2)
    # a batch: 2 hands × 2 docs, one page each (fg_threshold 0 keeps all tokens)
    pages = [_page(rng, 2) for _ in range(4)]
    hands = np.array(["A", "A", "B", "B"])
    docs = np.array(["a0", "a1", "b0", "b1"])

    z = batch_docvectors(model, netvlad, pages, num_class_tokens=1, patch_size=16,
                         fg_threshold=0.0, fg_method="contrast", embed_dim=dim,
                         max_tokens=32, max_windows=0, rng=rng, device="cpu")
    # same-hand-different-doc = positive, different-hand = negative
    pos = (hands[:, None] == hands[None, :]) & (docs[:, None] != docs[None, :])
    neg = hands[:, None] != hands[None, :]
    loss = masked_supcon(z, torch.from_numpy(pos), torch.from_numpy(neg))
    loss.backward()

    # the NetVLAD aggregation moved
    assert netvlad.centroids.grad is not None
    assert float(netvlad.centroids.grad.abs().sum()) > 0
    # ...AND the gradient reached the BACKBONE — this is what Tier-2 (frozen tokens)
    # could never do. Check a real ViT weight.
    pe = model.patch_embed.proj.weight
    assert pe.grad is not None
    assert float(pe.grad.abs().sum()) > 0


def test_token_and_window_caps_bound_the_input():
    # Caps must not break the graph: still a valid, grad-attached vector.
    model, netvlad, dim = _tiny_setup()
    rng = np.random.default_rng(3)
    v = page_docvector(model, netvlad, _page(rng, 8), num_class_tokens=1, patch_size=16,
                       fg_threshold=0.0, fg_method="contrast", embed_dim=dim,
                       max_tokens=16, max_windows=4, rng=rng, device="cpu")
    assert v.shape == (netvlad.centroids.shape[0] * dim,)
    assert v.requires_grad


def _synthetic_index():
    from pathlib import Path
    from mole.supervised.datasets import SupItem, SupervisedIndex
    idx = SupervisedIndex()
    for h in ["a/H0", "a/H1", "a/H2", "a/H3"]:          # 4 hands, 2 docs each, 1 page/doc
        for d in range(2):
            doc = f"{h}#d{d}"
            idx.items.append(SupItem(path=Path(f"{h.replace('/', '_')}_{d}.png"),
                                     archive="a", hand=h, doc=doc, confidence=None))
    return idx._reindex()


def test_train_joint_vlad_runs_and_selects(tmp_path):
    from mole.supervised.jointvlad import train_joint_vlad
    model, netvlad, dim = _tiny_setup()
    index = _synthetic_index()
    rng = np.random.default_rng(0)

    def load_crops(item):                                # disk-free: random crops per page
        return _page(rng, 2)

    fwd = dict(num_class_tokens=1, patch_size=16, fg_threshold=0.0, fg_method="contrast",
               embed_dim=dim, max_tokens=16, max_windows=0)
    model, netvlad, report = train_joint_vlad(
        model, netvlad, index, load_crops=load_crops, fwd=fwd,
        holdout_hands={"a/H2", "a/H3"}, epochs=2, lr=1e-3, device="cpu", seed=0,
        sampler_cfg=dict(hands_per_batch=2, docs_per_hand=2, batches_per_epoch=3),
        progress=False)

    assert report["epochs"] == 2
    assert report["best_epoch"] in (0, 1)
    assert 0.0 <= report["best_holdout_macro"] <= 1.0    # a real macro-mAP
    assert len(report["history"]) == 2
    assert report["history"][0]["loss"] is not None       # a batch with positives ran


def test_train_joint_vlad_ssl_runs_and_selects():
    # Label-free: two augmented views of a page = positive; different pages = negative.
    # Labels are used ONLY for the held-out selection metric.
    from mole.supervised.jointvlad import train_joint_vlad_ssl
    model, netvlad, dim = _tiny_setup()
    index = _synthetic_index()
    rng0 = np.random.default_rng(0)
    ssl_paths = [f"page_{i}.png" for i in range(8)]           # 8 unlabeled "pages"

    def load_view(path, rng):                                 # random windows = a random view
        return _page(rng0, 2)

    def load_crops(item):                                     # deterministic eval crops
        return _page(rng0, 2)

    fwd = dict(num_class_tokens=1, patch_size=16, fg_threshold=0.0, fg_method="contrast",
               embed_dim=dim, max_tokens=16, max_windows=0)
    model, netvlad, report = train_joint_vlad_ssl(
        model, netvlad, ssl_paths, load_view, holdout_index=index,
        holdout_hands={"a/H2", "a/H3"}, holdout_load_crops=load_crops, fwd=fwd,
        epochs=2, lr=1e-3, batch_pages=4, batches_per_epoch=3, device="cpu", seed=0,
        progress=False)

    assert report["objective"] == "self-supervised"
    assert report["epochs"] == 2 and report["best_epoch"] in (0, 1)
    assert 0.0 <= report["best_holdout_macro"] <= 1.0
    assert report["history"][0]["loss"] is not None           # a contrastive batch ran


def test_empty_foreground_is_graph_attached_not_nan():
    # A very high contrast threshold drops every token; the fallback zero token must
    # keep the vector finite and differentiable (grad flows via the centroids).
    model, netvlad, dim = _tiny_setup()
    rng = np.random.default_rng(4)
    v = page_docvector(model, netvlad, _page(rng, 2), num_class_tokens=1, patch_size=16,
                       fg_threshold=10.0, fg_method="contrast", embed_dim=dim,
                       max_tokens=0, max_windows=0, rng=rng, device="cpu")
    assert torch.isfinite(v).all()
    v.sum().backward()
    assert netvlad.centroids.grad is not None
