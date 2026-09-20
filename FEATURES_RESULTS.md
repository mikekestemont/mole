# Feature learning — the effect of training the ViT

The backbone is the biggest lever in the system. This consolidates the self-supervised
feature-learning results (previously scattered across run logs and `SUPERVISED_HANDOFF.md`,
which predates the final pooled run). Companion to `VLAD_ADAPTATION_RESULTS.md`, which covers
the *downstream* codebook/normalization levers.

**Backbone.** `vit_small`, AttMask self-supervised objective, **warm-started from Raven's
checkpoint** (arXiv:2409.00751, itself pretrained on Historical-WI), then finetuned in-domain on
the pooled charter corpus. All numbers: macro-mAP, VLAD-100, binarized + `--invert`, 224 px /
overlap 0, contrast foreground.

## raven-raw → in-domain SSL finetune

| archive (hands) | raven-raw (no in-domain training) | pooled SSL finetune (deployed) | Δ |
|---|---|---|---|
| Antwerp (13) | 0.718 | 0.817 | +0.099 |
| Utrecht (86) | 0.515 | 0.621 | +0.106 |
| Brackley (14) | 0.764 | 0.776 | +0.012 |
| Flanders (11) | 0.385 | 0.514 | +0.129 |
| Leroy (98) | 0.782 | 0.816 | +0.034 |
| **mean** | **0.633** | **0.709** | **+0.076** |

Deployed backbone: `runs/pooled_bin_ft/checkpoint.pth` (`vit_small@6ffcd327+step179100`).

## Conclusions

1. **Training the ViT is by far the biggest lever.** +0.076 mean (up to +0.13) dwarfs every
   downstream lever measured since: vocabulary adaptation +0.036, intra-normalization +0.020, SGR
   re-ranking +0.005, NetVLAD +0.006. The features carry the system; the codebook and aggregation
   tricks are refinements on top of them.

2. **Gain tracks in-domain data volume / headroom.** Largest on the hard, high-headroom
   collections (Flanders +0.129, Utrecht +0.106, Antwerp +0.099), tapering on near-ceiling Leroy
   (+0.034) and data-starved Brackley (+0.012). The earlier *solo* (per-collection) finetunes said
   the same thing by data volume: Utrecht (841 img) +0.137 > Antwerp (470) +0.109 > Brackley (300)
   +0.014 ≈ no-op.

3. **The generality tax vanished — deploy one model, not five.** A single pooled backbone over all
   five archives matched the per-collection specialists (gap closed to Antwerp −0.010, Brackley
   −0.002, Utrecht −0.030) *and* beat raven-raw everywhere. One general backbone is as good as five
   specialists — the result the search-engine/index goal needs.

4. **Training is done.** The pooled run plateaued by epochs 14–20 (+0.003..0.018), so the deployed
   checkpoint is settled; more SSL epochs are not the lever.

## Settled feature-side facts

- **`window_size=224` (token scale at embed) is the other dominant lever** — the single biggest
  knob after the backbone itself (e.g. Brackley 0.573 → 0.764 from the 512 → 224 change).
- **Codebook-free poolings underperform VLAD.** mean / meanstd / cov all sit below VLAD; a *more
  general* backbone *widens* the VLAD-vs-mean gap (pooled training spreads the token distribution
  into more modes, which VLAD's cluster-relative residuals exploit and a single mean averages away).
  So VLAD stays the aggregator; mean is not a viable incremental fallback on the general model.
- **HWI reproduction certified** — from-scratch Raven reproduction 0.795 vs paper 0.826, i.e. the
  pipeline is faithful (the residual gap is not a bug in feature extraction). See the parity
  ledger below (2026-09-20): every inference knob in the paper *and* in his released code has
  since been matched and the gap has not moved.
- **Polarity:** the model is white-on-black (Raven's regime); binarized input needs `--invert`.

## Layout cropping / the zone detector — MEASURED, and it's a null for retrieval (2026-08-23)

An earlier draft of this file called layout cropping "the biggest unrealised feature-side lever,"
extrapolating ~+0.05/archive from the fact that ground-truth cropping was worth **+0.053 macro on
Antwerp**. **That extrapolation was wrong**, and we now have the measurement (`scripts/run_zones_ab.sh`,
`outputs/zones_ab/`, fine-tuned `frag-obb-v3` detector, per archive: whole-page vs zone-restricted
windows, transductive codebook per arm):

| archive | off | on | Δmacro |
|---|---|---|---|
| Antwerp | 0.8171 | 0.8172 | +0.0001 (pre-cropped — N/A) |
| Brackley | 0.7758 | 0.7725 | −0.0033 |
| Flanders | 0.5109 | 0.5155 | +0.0046 |
| Leroy | 0.8126 | 0.8120 | −0.0006 |
| Utrecht | 0.6207 | 0.6263 | +0.0056 |
| **mean** | | | **+0.0013, CI [−0.008, +0.010] — indistinguishable from 0** |

**The detector does not move charter retrieval.** Hard archives (Flanders, Utrecht) lean marginally
positive, balanced ones marginally negative, all in the noise; guardrail passes (it doesn't hurt).
WHY: binarization + the contrast foreground filter already strip the background, so the detector's
extra crop is redundant — a little more background removed on hard archives, offset by mild clipping
on rare hands (the small-n hands wobble in both directions). The **+0.053 on Antwerp was
archive-specific** (its full pages carried unusually much removable background; `data/antwerp-bin`
is itself GT-cropped, so its A/B is a null by construction), NOT a general lever.

CONSEQUENCES: (1) charter writer-retrieval **does not need zones** — `use_zones=false` is the right
default, one fewer moving part. (2) The zone detector's value is **layout-dependent work** (the
scripy multi-column codices Lancelot/LTK191, per-column crops), not this pipeline. The detector
itself is strong (`frag-obb-v3`: mAP50 0.964, generalises to unseen multi-column layouts) — it just
isn't a retrieval lever here.

_Numbers are the canonical project results (some from runs of mid-2026); re-measurable from
`runs/pooled_bin_ft` and the archive datasets under the protocol above._

## Historical-WI parity ledger (raven checkpoint, 2026-07 → 2026-09-20)

Test split = 3600 pages / 720 writers, leave-one-out, cosine, PCA-whitening (384) fit on train.
Paper (arXiv:2409.00751, ICDAR 2024): **82.6 mAP / 91.9 Top-1** at S_eval = 56. Tim Raven confirmed
(2026-09-19/20) that the checkpoint we hold is the exact model and that the settings are right. His
released code is `Traven16/SSL_ViT_WR`; every row below matches the paper, the last also his code.

| run | what changed | mAP | Top-1 | Top-5 |
|---|---|---|---|---|
| hwi_w224 (July) | window 224, stride 224, raven fg, `--window-foreground`, k-means 10k×3 on all train tokens | 0.7953 | 0.9061 | 0.9319 |
| hwi_s56 (09-17) | + stride 56 (`overlap=0.75`), codebook/whitening reused | 0.8000 | 0.9056 | 0.9361 |
| **hwi_parity (09-19)** | + his codebook fit (`--kmeans-batch-size 1000000 --kmeans-n-init 1`, 20 % pool = `--vlad-max-descriptors 700000`), NO window pre-filter (commented out in his `keypoints.py`), whitening refit | **0.8009** | **0.9086** | 0.9358 |

Train smoke test (codebook + whitening fit on train, evaluated on train): 0.8902 / 0.9306.
Δ across the whole ledger is +0.6 mAP; the 2.5-point gap to the paper is untouched. Also ruled out on
the archives (`scripts/run_gmp_ab.py`, token cache): **GMP pooling** (`--vlad-pooling gmp`, the default of
his `VLAD` class but not of his inference script) — sum wins on Antwerp/Brackley at every gamma, mild
gain only on the skewed Flanders set; it is a per-cluster whitening of the residuals and fails the same
way PCA-whitening does on small archives.

Line-by-line diff of `mole.embed.vlad` / eval against his code (2026-09-19) left exactly one setting
unrun: `run.sh` uses `--stride_factor_eval=0.1` → **stride 22 px** for both the codebook features and
the test extraction (≈100× the windows of stride 224; ~15 h on one 2080 Ti). That run (`outputs/hwi_s22`)
is the last thing the released code can tell us; beyond it the residual is in an extraction whose settings
are not in the repo (`--extract_train=false` consumes pre-computed train features).

**Eval definition check vs the competition paper** (Fiel et al., ICDAR 2017, §IV): mole computes
AP = Σ_k P(k)·rel(k) / |relevant| with the query removed from the gallery (4 relevant per HWI query),
Top-1 = first ranked page by the same writer, Top-k soft = any hit in the first k. Raven's
`retrieval.py` is the same (self set to −∞, dropped from the ranking, AP over the relevant positions),
so paper and ledger are on one definition. The competition text itself defines AveP over *n = 3600* with
*5* relevant documents, i.e. with the query inside its own ranking — that convention scores higher
(≈ (1 + 4·AP)/5 for a page ranked first by itself), so 2017 competition numbers (Tébessa II 55.6) are
not directly comparable to leave-one-out mAP. mole does not report the competition's hard-k / p@k.

