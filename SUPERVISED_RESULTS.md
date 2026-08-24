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
| Joint differentiable VLAD (backbone + NetVLAD) | features *through* the aggregator | **positive** — Antwerp fold +0.038 (LOAO), the best signal yet — see below |

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

## Joint differentiable VLAD — positive (2026-08-23, first fold)

`src/mole/supervised/jointvlad.py` + `scripts/run_joint_vlad.py`: a masked-SupCon loss on page
document-vectors backprops into both the ViT and the NetVLAD centroids. Rationale: capture the +0.015
supervised signal **without** the compression tax — move the *features* through the aggregator rather than
project a fixed 38,400-d vector down.

**Antwerp fold (held out, trained on the other four):** frozen **0.7214** → trained **0.7598** =
**+0.0384**, LOAO — genuine generalization.

- **Beats post-aggregation supervision (+0.015) and pays no compression tax** — the full descriptor, so
  this is a real accuracy gain, not a 128-d parity trade. Matches vocabulary adaptation (+0.036). The
  best supervised signal in the project.
- **The literature-faithful NetVLAD working as advertised** — gradients into the backbone, which the
  frozen Tier-2 version (+0.006) structurally could not do.

⚠️ **Learning rate is everything.** `lr 1e-4` **destroys** the pretrained features (0.72 → 0.11 ≈ chance
in one epoch; the training loss falls while held-out retrieval collapses — a wrecking ball on a strong
backbone). `lr 1e-6` is stable and **still climbing** at epoch 5 (best = last), so +0.038 is a **floor**.

### Confirmed at full window (2026-08-23, `scripts/run_joint_eval.py` + standard `mole eval`)

The in-loop proxy (8-window cap) inflated the gain ~2×; the full-window, standard `--cross-doc-only` LOAO
verdict is smaller but real:

| archive | frozen | trained | Δmacro |
|---|---|---|---|
| Antwerp | 0.7995 | 0.8145 | +0.0149 |
| Brackley | 0.7629 | 0.7557 | −0.0071 |
| Flanders | 0.4159 | 0.5044 | **+0.0885** |
| Leroy | 0.8164 | 0.8141 | −0.0024 |
| Utrecht | 0.5859 | 0.5892 | +0.0034 |
| **mean** | | | **+0.0195** (CI [+0.010, +0.030], REAL, guardrail passed) |

- **The +0.043 proxy was the 8-window artifact** — Brackley's frozen baseline is 0.763 at full window,
  not the 0.26 the capped proxy showed.
- **α is faithful** — frozen-NetVLAD absolutes match the known hard-VLAD numbers, so `alpha 0.5` was never
  the bottleneck (no fix needed).
- **Real but modest, and almost entirely Flanders** (+0.088; others flat-to-slightly-negative;
  hand-weighted only +0.007). Still climbing at 30 epochs, so a longer run might reach ~+0.025–0.03.

**Verdict:** the literature-faithful NetVLAD works — the first lever to show supervision helping on the
*full* descriptor (no compression tax) — but it lands **mid-pack and is the most expensive lever**, below
vocabulary adaptation (+0.036, cheap, no retraining) which wins on both accuracy and cost. Worth pushing
(longer run) only if **Flanders specifically** matters, where its +0.088 is the standout. Otherwise the
backbone remains the dominant lever and adaptation the best downstream one.
