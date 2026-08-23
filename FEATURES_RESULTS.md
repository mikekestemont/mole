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
  pipeline is faithful (the residual gap is not a bug in feature extraction).
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
