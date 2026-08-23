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
