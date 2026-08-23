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

## The biggest remaining feature-side lever (not yet realised)

**Layout cropping / the zone detector.** Ground-truth layout cropping is worth **+0.053 macro on
Antwerp** — and Antwerp is the *only* archive with it applied; the other four run `use_zones=false`
and are plausibly leaving ~0.05 each on the table. This is a *feature-quality* lever (it controls
what pixels the ViT sees per window), and it is arguably bigger than anything in the VLAD thread.
The in-domain OBB zone detector (`frag-obb-v2`, 342 pages) is built and integrated into `mole prep
--yolo-weights`; the open step is to crop the four archives with it, re-embed, and measure
(`scripts/train_zone_obb.py`, `scripts/ls_to_yolo.py`).

_Numbers are the canonical project results (some from runs of mid-2026); re-measurable from
`runs/pooled_bin_ft` and the archive datasets under the protocol above._
