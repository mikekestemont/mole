"""Masked SupCon loss + metric-learning trainers for the supervised module.

The loss (:func:`masked_supcon`) is the departure from stock SupCon that makes
the negative rule safe under partial labels: the denominator sums over
*confirmed* pairs only — the positives plus the confirmed negatives from
:func:`mole.supervised.datasets.pair_masks` — never "every non-positive". A
same-document pair, or anything touching an unlabeled window, is neither a
positive nor a negative and so contributes to neither the numerator nor the
denominator.

``train_metric`` (the Tier-1 head trainer on cached window descriptors, and the
Tier-3 hybrid) lands in Phase 2B — it needs the feature cache to exist.
"""

from __future__ import annotations

from pathlib import Path


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

    # log-sum-exp over the confirmed denominator, numerically stabilised.
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


def train_metric(config_path: str | Path, base_checkpoint: str | Path,
                 labels_root: str | Path, output_dir: str | Path | None = None):
    """Train the Tier-1 head (or Tier-3 hybrid) — Phase 2B (needs the cache)."""
    raise NotImplementedError(
        "train_metric lands in Phase 2B: Tier-1 head on the feature cache "
        "(sup.tier: head) / Tier-3 hybrid (sup.tier: hybrid). The loss "
        "(masked_supcon) and sampler (HandBatchSampler) are ready.")
