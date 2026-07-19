"""Masked SupCon loss + the Tier-1 head trainer.

The loss (:func:`masked_supcon`) is the departure from stock SupCon that makes
the negative rule safe under partial labels: the denominator sums over
*confirmed* pairs only — the positives plus the confirmed negatives from
:func:`mole.supervised.datasets.pair_masks` — never "every non-positive". A
same-document pair, or anything touching an unlabeled window, is neither a
positive nor a negative and so contributes to neither the numerator nor the
denominator.

:func:`train_head` learns a small projection on cached window descriptors (the
Tier-1 head): no per-hand parameters, model-selected on **held-out-hand**
retrieval macro-mAP so memorizing a starved hand buys nothing. :func:`train_metric`
is the config-driven entry point (``sup.tier: head`` now; ``hybrid`` is Phase 5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def masked_supcon(z, pos_mask, neg_mask, temperature: float = 0.07):
    """Supervised contrastive loss over explicit positive / negative masks.

    ``z`` — ``[B, d]`` embeddings (L2-normalised internally). ``pos_mask`` and
    ``neg_mask`` — ``[B, B]`` boolean, from
    :func:`mole.supervised.datasets.pair_masks`; they are disjoint and both
    exclude the diagonal.

    For anchor ``i`` with positive set ``P(i)`` and confirmed-negative set
    ``N(i)``::

        L_i = -1/|P(i)| Σ_{p∈P(i)} log[ exp(z_i·z_p/τ)
                                        / Σ_{a∈P(i)∪N(i)} exp(z_i·z_a/τ) ]

    The denominator ranges over ``P(i)∪N(i)`` only — the stock-SupCon choice of
    "all a≠i" is precisely what would turn an unlabeled or same-document pair
    into an implicit negative. Anchors with ``|P(i)|=0`` are dropped from the
    mean. Returns a scalar tensor (0, still attached to the graph, if no anchor
    has a positive).
    """
    import torch

    z = torch.nn.functional.normalize(z, dim=1)
    logits = (z @ z.t()) / temperature                 # [B, B]

    pos = pos_mask.bool()
    neg = neg_mask.bool()
    denom = pos | neg                                  # confirmed pairs only

    neg_inf = torch.finfo(logits.dtype).min
    row_max = logits.masked_fill(~denom, neg_inf).max(dim=1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max,
                          torch.zeros_like(row_max))    # rows with empty denom
    exp = torch.exp(logits - row_max) * denom.to(logits.dtype)
    log_denom = torch.log(exp.sum(dim=1, keepdim=True) + 1e-12) + row_max
    log_prob = logits - log_denom                       # [B, B]

    pos_f = pos.to(logits.dtype)
    pos_count = pos_f.sum(dim=1)                         # [B]
    valid = pos_count > 0
    if not bool(valid.any()):
        return z.sum() * 0.0                            # keep the graph, zero loss
    per_anchor = (pos_f * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -per_anchor.mean()


# ---------------------------------------------------------------------- the head
def build_head(kind: str, in_dim: int, out_dim: int):
    """The projection head: ``linear`` (v0) or a 2-layer ``mlp`` (v1)."""
    import torch.nn as nn

    if kind == "linear":
        return nn.Linear(in_dim, out_dim)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
                             nn.Linear(in_dim, out_dim))
    raise ValueError(f"head kind must be 'linear' or 'mlp', got {kind!r}")


def _l2(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _doc_embeddings(cache, head, device):
    """Per-document embedding = L2(mean of projected window descriptors)."""
    import torch

    from collections import defaultdict

    X = torch.from_numpy(cache.descriptors).to(device)
    with torch.no_grad():
        Z = torch.nn.functional.normalize(head(X), dim=1).cpu().numpy()
    by_doc: dict[str, list[int]] = defaultdict(list)
    doc_hand: dict[str, str] = {}
    for i, (d, h) in enumerate(zip(cache.window_doc, cache.window_hand)):
        by_doc[d].append(i)
        doc_hand[d] = h
    docs = list(by_doc)
    emb = np.stack([_l2(Z[by_doc[d]].mean(0)) for d in docs]) if docs \
        else np.zeros((0, 1), np.float32)
    labels = np.asarray([doc_hand[d] for d in docs], dtype=object)
    return emb, labels


def holdout_macro_map(holdout_cache, head, device) -> float:
    """Cross-document macro-mAP over held-out hands (the fast model-select proxy)."""
    from mole.eval.retrieval import _rank_metrics, _similarity

    emb, labels = _doc_embeddings(holdout_cache, head, device)
    if len(emb) < 2:
        return 0.0
    sim = _similarity(emb.astype(np.float64), "cosine")
    off = ~np.eye(len(emb), dtype=bool)
    scores = _rank_metrics(sim, labels, off, (1,))
    return float(scores.macro_map) if scores else 0.0


def train_head(cache, *, holdout_hands: set[str], out_dim: int = 128,
               temperature: float = 0.07, kind: str = "linear",
               sampler_cfg: dict | None = None, seed: int = 0, epochs: int = 30,
               lr: float = 1e-3, weight_decay: float = 1e-4,
               device: str | None = None, progress: bool = True):
    """Train the projection head on a :class:`FeatureCache`; return ``(head, report)``.

    Trains on TRAIN-hand windows only (all labeled hands in the cache that are not
    in ``holdout_hands``); the held-out hands are unseen classes and the model is
    selected on their cross-document macro-mAP, so overfitting a starved train
    hand cannot improve the stopping metric. The split itself is the caller's
    responsibility (``SupervisedIndex.split_hands`` / ``write_holdout_split``).
    """
    import torch

    from mole.supervised.datasets import HandBatchSampler, pair_masks

    dev = torch.device(device) if device else torch.device("cpu")
    in_dim = cache.dim

    holdout_hands = set(holdout_hands)
    all_hands = {h for h in cache.window_hand if h}
    train_hands = all_hands - holdout_hands

    train_cache = cache.filter(train_hands)
    holdout_cache = cache.filter(holdout_hands)

    head = build_head(kind, in_dim, out_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    X = torch.from_numpy(train_cache.descriptors).to(dev)
    sampler = HandBatchSampler(train_cache, seed=seed, **(sampler_cfg or {}))

    best_macro, best_state, history = -1.0, None, []
    from mole.progress import track
    for ep in track(range(epochs), "Training head", unit="epoch", disable=not progress):
        head.train()
        losses = []
        for rows, hands, docs in sampler:
            pos, neg = pair_masks(hands, docs)
            z = head(X[rows])
            loss = masked_supcon(z, torch.from_numpy(pos).to(dev),
                                 torch.from_numpy(neg).to(dev), temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        head.eval()
        macro = holdout_macro_map(holdout_cache, head, dev)
        history.append({"epoch": ep, "loss": float(np.mean(losses)) if losses else 0.0,
                        "holdout_macro": macro})
        if macro > best_macro:
            best_macro = macro
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    report = {
        "best_holdout_macro": best_macro, "in_dim": in_dim, "out_dim": out_dim,
        "kind": kind, "temperature": temperature, "epochs": epochs, "seed": seed,
        "base_model_id": cache.meta.get("model_id"),
        "n_train_hands": len(train_hands), "n_holdout_hands": len(holdout_hands),
        "train_hands": sorted(train_hands), "holdout_hands": sorted(holdout_hands),
        "history": history,
    }
    return head, report


_SUP_DEFAULTS = {
    "tier": "head", "head": "linear", "out_dim": 128, "temperature": 0.07,
    "min_confidence": None, "holdout_frac": 0.2, "seed": 0, "epochs": 30,
    "lr": 1e-3, "weight_decay": 1e-4, "cache_dir": None,
    "sampler": {"hands_per_batch": 16, "docs_per_hand": 2, "windows_per_doc": 4,
                "same_archive_frac": 0.5},
}


def train_metric(config_path: str | Path, base_checkpoint: str | Path,
                 labels_root: str | Path, output_dir: str | Path | None = None):
    """Config-driven Tier-1 head training (``sup.tier: head``).

    Builds (or reuses) the feature cache from ``base_checkpoint``, trains the head
    on the pooled labels under ``labels_root``, and writes ``head.pt`` +
    ``report.json`` + the frozen hand split. ``sup.tier: hybrid`` (Tier 3) is
    Phase 5.
    """
    import torch

    from mole.config import config_hash, load_config
    from mole.supervised.datasets import (
        FeatureCache, build_feature_cache, load_labeled_pairs)

    cfg = load_config(config_path)
    sup = {**_SUP_DEFAULTS, **(cfg.get("sup") or {})}
    if sup["tier"] != "head":
        raise NotImplementedError(
            f"sup.tier={sup['tier']!r}: only 'head' is implemented (hybrid = Phase 5)")

    out = Path(output_dir or "runs/sup_head")
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(sup["cache_dir"]) if sup["cache_dir"] else out / "cache"

    index = load_labeled_pairs(labels_root, sup["min_confidence"])
    if (cache_dir / "cache.npy").is_file():
        cache = FeatureCache.load(cache_dir)
        print(f"[mole] reusing feature cache at {cache_dir} ({cache.n_windows:,} windows)")
    else:
        cache = build_feature_cache(base_checkpoint, index, cache_dir)

    # freeze the split so every tier / rerun uses the identical held-out hands.
    _, hold = index.write_holdout_split(out / "split.json",
                                        holdout_frac=sup["holdout_frac"], seed=sup["seed"])
    head, report = train_head(
        cache, holdout_hands=set(hold.hands), out_dim=sup["out_dim"],
        temperature=sup["temperature"], kind=sup["head"], sampler_cfg=sup["sampler"],
        seed=sup["seed"], epochs=sup["epochs"], lr=sup["lr"],
        weight_decay=sup["weight_decay"])

    head_path = out / "head.pt"
    torch.save({
        "state_dict": head.state_dict(), "in_dim": report["in_dim"],
        "out_dim": report["out_dim"], "kind": report["kind"],
        "base_model_id": cache.meta.get("model_id"),
        "config_hash": config_hash(cfg),
    }, head_path)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[mole] ✓ head → {head_path}  (held-out macro-mAP {report['best_holdout_macro']:.4f})")
    return head_path
