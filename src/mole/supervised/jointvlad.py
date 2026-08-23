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

from mole.supervised.metric import masked_supcon


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


# --------------------------------------------------------------- page-level sampler
def _doc_pages(index) -> dict[str, list[int]]:
    """Map each namespaced document → the item indices (pages) that belong to it."""
    from collections import defaultdict
    m: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(index.items):
        m[it.doc].append(i)
    return dict(m)


class PageBatchSampler:
    """Yield P hands × D documents of PAGES (one page sampled per document).

    Same contract as the window-level :class:`HandBatchSampler`, but the unit is a
    page (retrieval ranks pages). Only hands with ≥ ``docs_per_hand`` distinct
    documents can anchor a batch (a positive needs two documents of one hand).
    ``same_archive_frac`` biases the negative hands toward the anchors' archives.
    """

    def __init__(self, index, *, hands_per_batch: int = 8, docs_per_hand: int = 2,
                 batches_per_epoch: int = 100, same_archive_frac: float = 0.5,
                 exclude_hands: set[str] | None = None, seed: int = 0):
        self.index = index
        self.doc_pages = _doc_pages(index)
        self.P, self.D = hands_per_batch, docs_per_hand
        self.batches = batches_per_epoch
        self.same_frac = same_archive_frac
        self._rng = np.random.default_rng(seed)
        ex = exclude_hands or set()
        # anchorable = enough distinct docs, not held out
        self.hands = [h for h, docs in index.docs_by_hand.items()
                      if len(docs) >= docs_per_hand and h not in ex]

    def __iter__(self):
        for _ in range(self.batches):
            if len(self.hands) < 1:
                return
            k = min(self.P, len(self.hands))
            hands = list(self._rng.choice(self.hands, k, replace=False))
            items, out_hands, out_docs = [], [], []
            for h in hands:
                docs = list(self.index.docs_by_hand[h])
                pick = self._rng.choice(docs, min(self.D, len(docs)), replace=False)
                for d in pick:
                    pages = self.doc_pages[d]
                    items.append(int(self._rng.choice(pages)))
                    out_hands.append(h)
                    out_docs.append(d)
            yield items, out_hands, out_docs


# ------------------------------------------------------------ page-level LOAO eval
def holdout_doc_macro_map(model, netvlad, index, hands_subset, load_crops, fwd, device):
    """Cross-document macro-mAP over ``hands_subset`` — the model-selection proxy.

    Per-document embedding = L2(mean of its pages' NetVLAD vectors); macro-mAP is
    then averaged over hands, exactly as the frozen-head selector does, so the
    number is comparable to every other lever's held-out macro-mAP.
    """
    import torch

    from collections import defaultdict

    from mole.eval.retrieval import _rank_metrics, _similarity

    rng = np.random.default_rng(0)
    idxs = [i for i, it in enumerate(index.items) if it.hand in hands_subset]
    if len(idxs) < 2:
        return 0.0
    was_training = model.training
    model.eval(); netvlad.eval()
    by_doc: dict[str, list[np.ndarray]] = defaultdict(list)
    doc_hand: dict[str, str] = {}
    with torch.no_grad():
        for i in idxs:
            it = index.items[i]
            v = page_docvector(model, netvlad, load_crops(it), rng=rng, device=device, **fwd)
            by_doc[it.doc].append(v.cpu().numpy())
            doc_hand[it.doc] = it.hand
    if was_training:
        model.train(); netvlad.train()
    docs = list(by_doc)
    emb = np.stack([_l2(np.mean(by_doc[d], axis=0)) for d in docs])
    labels = np.asarray([doc_hand[d] for d in docs], dtype=object)
    sim = _similarity(emb.astype(np.float64), "cosine")
    off = ~np.eye(len(emb), dtype=bool)
    scores = _rank_metrics(sim, labels, off, (1,))
    return float(scores.macro_map) if scores else 0.0


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-12 else v.astype(np.float32)


# ------------------------------------------------------------------- training loop
def train_joint_vlad(model, netvlad, index, *, load_crops, fwd, holdout_hands: set[str],
                     epochs: int = 20, lr: float = 1e-4, weight_decay: float = 0.05,
                     sampler_cfg: dict | None = None, temperature: float = 0.07,
                     device: str = "cpu", seed: int = 0, progress: bool = True):
    """Joint finetune the backbone + NetVLAD by masked-SupCon on page doc-vectors.

    Warm-started weights are the caller's responsibility (load the pooled backbone,
    ``NetVLAD.from_codebook``); this runs the loop and model-selects the best epoch on
    the held-out-hand macro-mAP (``holdout_hands`` excluded from the sampler AND used
    for selection). Returns ``(model, netvlad, report)`` with the best weights loaded.
    """
    import torch

    from mole.progress import track
    from mole.supervised.datasets import pair_masks

    dev = torch.device(device)
    model.to(dev).train(); netvlad.to(dev).train()
    params = [p for p in list(model.parameters()) + list(netvlad.parameters())
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sampler = PageBatchSampler(index, exclude_hands=holdout_hands, seed=seed,
                               **(sampler_cfg or {}))
    rng = np.random.default_rng(seed)

    best_macro, best_epoch, best_state, history = -1.0, -1, None, []
    for ep in range(epochs):
        model.train(); netvlad.train()
        losses = []
        for items, hands, docs in track(sampler, f"epoch {ep + 1}/{epochs}",
                                        total=sampler.batches, unit="batch", disable=not progress):
            crops = [load_crops(index.items[i]) for i in items]
            z = batch_docvectors(model, netvlad, crops, rng=rng, device=dev, **fwd)
            pos, neg = pair_masks(hands, docs)
            if not pos.any():                             # no positive in this batch
                continue
            loss = masked_supcon(z, torch.from_numpy(pos).to(dev),
                                 torch.from_numpy(neg).to(dev), temperature=temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        macro = holdout_doc_macro_map(model, netvlad, index, holdout_hands, load_crops, fwd, dev)
        loss_mean = float(np.mean(losses)) if losses else None
        history.append({"epoch": ep, "loss": loss_mean, "holdout_macro": macro})
        if progress:
            lt = f"{loss_mean:.4f}" if loss_mean is not None else "—"
            print(f"[joint] epoch {ep + 1}/{epochs}: loss {lt}  holdout-macro {macro:.4f}"
                  + ("  ← best" if macro > best_macro else ""), flush=True)
        if macro > best_macro:
            best_macro, best_epoch = macro, ep
            best_state = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in netvlad.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state[0]); netvlad.load_state_dict(best_state[1])
    report = {"best_holdout_macro": best_macro, "best_epoch": best_epoch,
              "epochs": epochs, "history": history}
    return model, netvlad, report
