# Supervised metric-learning results

Where labels help — and where they don't. Companion to `FEATURES_RESULTS.md` (backbone)
and `VLAD_ADAPTATION_RESULTS.md` (codebook levers). The recurring finding: supervision
*transfers across collections*, but the aggregator decides whether it survives.

## The ladder so far

| tier | what it supervises | verdict |
|---|---|---|
| Tier-1 (head on window descriptors) | pre-aggregation | **fails under VLAD** (−0.013 LOAO); +0.066 under mean — the aggregator discards it |
| Tier-2 (NetVLAD on a frozen token cache) | the aggregation, backbone frozen | **null** (+0.006) — ≈ hard VLAD by construction; literature trains it jointly |
| Post-aggregation metric learning | the finished VLAD doc vector | **real but small** (+0.015 over whitening) — see below |
| Joint differentiable VLAD (backbone + NetVLAD) | features *through* the aggregator | **built, not yet measured** — the literature-faithful test |

## Post-aggregation metric learning (2026-08-23)

`scripts/run_doc_metric.py` on the **universal-codebook** embeddings (`outputs/universal_full/*.npy` —
one shared space; per-archive transductive spaces are not comparable), leave-one-archive-out. Three
spaces at `out_dim=128`, all fit on the *other four* archives:

- **raw** — the 38,400-d universal VLAD vector, unprojected.
- **pca** — PCA-whitened to 128-d (a strong unsupervised trick; the control that matters).
- **supervised** — whiten→`whiten_dim` then a masked-SupCon projection to 128-d.

Supervision only earns credit for **sup − pca** at the same output dim (beating raw is just whitening
wearing a medal). Honest run (fixed `whiten_dim=512`, no per-archive tuning):

| archive | raw | pca | supervised | Δ sup−pca | Δ sup−raw |
|---|---|---|---|---|---|
| Antwerp | 0.7996 | 0.7902 | 0.8050 | +0.0147 | +0.0054 |
| Brackley | 0.7625 | 0.7665 | 0.7603 | −0.0062 | −0.0022 |
| Flanders | 0.4156 | 0.3899 | 0.4398 | **+0.0499** | +0.0242 |
| Leroy | 0.8131 | 0.7621 | 0.7699 | +0.0079 | −0.0431 |
| Utrecht | 0.5839 | 0.5095 | 0.5179 | +0.0084 | −0.0660 |
| **mean** | | | | **+0.0149** | **−0.0164** |

(An optimistic sweep keeping the best `whiten_dim` per archive gave +0.0248 vs pca — so ~60% of the
gain survives removing the tuning; the effect is real, the swept number was ~40% peeking inflation.)

**Two conclusions:**

1. **Supervision is real but small** — +0.015 over its fair control, positive on 4/5, concentrated on
   **Flanders (+0.050)**: the same "signal lives in the hard collections" pattern as adaptation and
   intra-norm. The first non-null supervised result in the project.
2. **It's a compression win, not an accuracy win.** The supervised 128-d is still −0.016 *below* the full
   38,400-d raw VLAD (whitening loses −0.031, supervision recovers +0.015). So it does **not** beat the
   deployed descriptor on accuracy — but a supervised 128-d at near-parity, **300× smaller**, in one
   shared cross-archive space that *generalizes* (trained on four archives, tested on the fifth), is a
   real **deployment** win for a large index (storage/latency), independent of accuracy.

## Joint differentiable VLAD — pending

Built and runnable (`src/mole/supervised/jointvlad.py`, `scripts/run_joint_vlad.py`; a masked-SupCon
loss on page doc-vectors backprops into both the ViT and the NetVLAD centroids). Its rationale, given the
above: capture the +0.015 supervised signal **without** the compression tax — by moving the *features*
through the aggregator rather than projecting a fixed 38,400-d vector down. Weak-green motivation: there
is learnable signal, but the ceiling looks modest. If it flatlines like Tier-1/2, the frozen features are
the ceiling and the answer is to ship the current descriptor. First fold (Antwerp) is the deciding read.
