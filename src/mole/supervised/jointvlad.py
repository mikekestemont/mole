"""Fully differentiable VLAD: joint backbone + NetVLAD finetuning.

Tier 2 (``netvlad.py``) trained the NetVLAD layer over a FROZEN token cache — and
that is the regime where NetVLAD ≈ hard VLAD by construction (the literature's
NetVLAD gains come from training the aggregation JOINTLY with the backbone, not as
a post-hoc bolt-on; measured null, commit 36509be). This module closes that gap:
it runs each page's foreground tokens through the LIVE ViT, aggregates them with
the differentiable :class:`~mole.supervised.netvlad.NetVLAD` layer, and lets a
metric loss on the resulting document vectors backprop **into the backbone**.

The unit is a PAGE (retrieval ranks pages). The memory cost of backprop through a
whole page's windows × K clusters is the constraint, so two caps bound it:
``max_windows`` (windows per page actually pushed through the ViT) and
``max_tokens`` (foreground tokens fed to NetVLAD). Power-norm + L2 make the VLAD
descriptor ~invariant to token count, so subsampling is sampling noise, not a
train/deploy mismatch (the Tier-2 design property).

This file holds the differentiable forward; the training loop, page sampler and
CLI build on it.
"""

from __future__ import annotations

import numpy as np


def page_docvector(model, netvlad, crops, *, num_class_tokens: int, patch_size: int,
                   fg_threshold: float, fg_method: str, embed_dim: int,
                   max_tokens: int, max_windows: int, rng, device):
    """One page's window crops → its NetVLAD document vector, gradients intact.

    ``crops`` is a list of ``[C, S, S]`` tensors (deterministic window resize, as at
    embed time). Returns a ``[K*embed_dim]`` tensor still attached to the ViT and the
    NetVLAD parameters, so ``loss.backward()`` reaches both. ``max_windows``/``max_tokens``
    bound backprop memory (0 = uncapped). A page with no foreground falls back to a
    single zero token, which keeps the vector graph-attached (grad defined, not NaN).
    """
    import torch

    from mole.embed.extract import _foreground_mask
    from mole.embed.pooling import patch_descriptors

    if max_windows and len(crops) > max_windows:          # cap ViT forwards per page
        pick = rng.choice(len(crops), max_windows, replace=False)
        crops = [crops[i] for i in pick]

    batch = torch.stack(crops).to(device)                 # [W, C, S, S]
    tokens = model(batch, return_attention=False, return_all_tokens=True)  # [W, seq, dim], GRAD
    patches = patch_descriptors(tokens, num_class_tokens)                   # [W, num_patches, dim]
    keep = _foreground_mask(crops, patch_size, fg_threshold, method=fg_method)  # [W, num_patches] bool
    fg = patches[keep.to(patches.device)]                 # [T, dim] foreground tokens (GRAD)

    if fg.shape[0] == 0:                                   # no foreground → graph-attached zero token
        fg = torch.zeros(1, embed_dim, device=device)
    elif max_tokens and fg.shape[0] > max_tokens:         # subsample tokens into NetVLAD
        idx = torch.from_numpy(rng.choice(fg.shape[0], max_tokens, replace=False)).to(device)
        fg = fg[idx]

    return netvlad(fg)                                    # [K*dim] doc vector (GRAD)


def batch_docvectors(model, netvlad, pages_crops, *, num_class_tokens: int,
                     patch_size: int, fg_threshold: float, fg_method: str,
                     embed_dim: int, max_tokens: int, max_windows: int, rng, device):
    """Document vectors for a batch of pages → ``[B, K*embed_dim]`` (gradients intact).

    Pages are encoded one at a time (each is a variable number of windows/tokens) and
    stacked, so a batch shares one backward through the ViT and NetVLAD.
    """
    import torch

    vecs = [page_docvector(model, netvlad, crops, num_class_tokens=num_class_tokens,
                           patch_size=patch_size, fg_threshold=fg_threshold,
                           fg_method=fg_method, embed_dim=embed_dim, max_tokens=max_tokens,
                           max_windows=max_windows, rng=rng, device=device)
            for crops in pages_crops]
    return torch.stack(vecs)                               # [B, K*dim]
