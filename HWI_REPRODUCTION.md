# Historical-WI reproduction of Raven, Matei & Fink (ICDAR 2024) — the record

**Claim under test.** arXiv:2409.00751 reports **82.6 mAP / 91.9 Top-1** on the ICDAR 2017
Historical-WI test set (3600 pages, 720 writers × 5) for the AttMask ViT-S/16 + VLAD (K=100) +
PCA-whitening (384) pipeline, with the codebook and PCA "fitted on the training set" and
S_eval = 56. We hold the exact checkpoint (confirmed by the author, 2026-09-20).

**Outcome (2026-09-20).** Every setting stated in the paper *or* found in the released code
(`Traven16/SSL_ViT_WR`) was matched; fit-on-train never exceeds **0.8015 mAP** at any stride.
Fitting the codebook and the PCA-whitening **on the test set** gives **0.8226 / 0.9156** at
stride 224, within the stride-56 increment (~+0.5) of the published number. The published result
is consistent with a transductive fit on the test split, not with the stated protocol.
Reproduction closed; nothing downstream changes (mole indexes archives transductively anyway).

## Results

All rows: raven checkpoint, native binarized HWI (white-on-black, `--no-invert`), window 224,
`--foreground-method raven` (t_fg = 10 px per 16×16 patch), plain VLAD (`--no-vlad-intra-norm`,
sum pooling), PCA-whitening to 384 dims, cosine, leave-one-out mAP with the query removed from
the gallery (same definition as the author's `retrieval.py`; see FEATURES_RESULTS.md for the
2017 competition's different convention).

| run (server `outputs/…`) | codebook + PCA fit on | test stride | k-means | window pre-filter | mAP | Top-1 | Top-5 |
|---|---|---|---|---|---|---|---|
| `hwi_w224` (2026-07-16) | train, all 3.47 M tokens | 224 | 10 k × 3 | 2.5 % | 0.7953 | 0.9061 | 0.9319 |
| `hwi_s56` (2026-09-17) | train (reused from above) | 56 | 10 k × 3 | 2.5 % | 0.8000 | 0.9056 | 0.9361 |
| `hwi_parity` (2026-09-19) | train, 20 % sample (700 k) | 56 | 1 M × 1 | none | 0.8009 | 0.9086 | 0.9358 |
| `hwi_s22` (2026-09-20) | train, stride-22 features, 4 M reservoir of 334 M | 22 | 1 M × 1 | none | 0.8015 | 0.9075 | 0.9361 |
| **`hwi_trans`** (2026-09-20) | **test**, 4 M reservoir of 10.6 M | 224 | 1 M × 1 | none | **0.8226** | **0.9156** | **0.9411** |
| paper | "train" | 56 | minibatch | 2.5 % | 0.826 | 0.919 | — |

Train-set smoke checks (fit on train, evaluated on train): 0.8902 / 0.9306 (stride 224),
0.8944 / 0.9332 (stride 22).

Reading: the four fit-on-train rows span 0.6 mAP across every inference-side change
(stride 224 → 56 → 22, k-means batch, codebook pool, window filter). The single change of
fitting on the test set is worth +2.7 mAP / +1.0 Top-1 and closes the gap.

Also tested and negative: **GMP pooling** (the default of the author's `VLAD` class, `gamma=1000`,
not used by his inference script) on the four charter archives via `scripts/run_gmp_ab.py` — sum
pooling wins at every γ ∈ {10, 100, 1000, 10000} on Antwerp (0.815 vs 0.710–0.795) and Brackley
(0.767 vs 0.732–0.760); only the skewed Flanders set gains (+0.024 at γ = 10000).

## What was diffed against the released code

`aggregators.py`, `pipeline.py`, `retrieval.py`, `keypoints.py`, `features.py`, `models_vit.py`,
`attmask/main_attmask.py` (2026-09-19). Identical: ToTensor without ImageNet normalisation
(`morph` augmentation default), fg-token rule `avg_pool(16) ≥ 10/256`, KD-tree / argmin hard
assignment, residual sum, signed sqrt power-norm, global L2, `PCA(384, whiten=True)` fit and
transform (exact SVD here vs sklearn's randomized), cosine ranking, AP over relevant positions
with self removed, soft Top-k; no per-image descriptor cap on either side (`PerPageKeypointSampler`
only in SIFT mode, `SampleSubsetOfExtractedLocalFeatures` never instantiated); `Retrieval.rerank`
stored, never used. Matched by flag: `MiniBatchKMeans(100, n_init=1, batch_size=1_000_000)`
(`--kmeans-batch-size 1000000 --kmeans-n-init 1`), 20 % per-page codebook sample
(`--vlad-max-descriptors 700000`), no window pre-filter (`WhitenessFilter(0.025)` is commented
out), `run.sh`'s `--stride_factor_eval=0.1` → 22 px (`--set overlap=0.9018`).

## Commands (server, conda env `mole`, one RTX 2080 Ti)

```bash
# hwi_parity — paper protocol + the release's codebook fit
mole embed checkpoints/raven_checkpoint.pth data/hwi-train outputs/hwi_parity/train.npy --pooling vlad \
  --kmeans-batch-size 1000000 --kmeans-n-init 1 --vlad-max-descriptors 700000 --whiten-dim 384 \
  --foreground --foreground-method raven --no-vlad-intra-norm --no-invert \
  --set window_size=224 --set overlap=0 --set use_zones=false
mole embed checkpoints/raven_checkpoint.pth data/hwi-test outputs/hwi_parity/test.npy --pooling vlad \
  --codebook-from outputs/hwi_parity/train.codebook.npy --whiten-from outputs/hwi_parity/train.whiten.npz \
  --foreground --foreground-method raven --no-vlad-intra-norm --no-invert \
  --set window_size=224 --set overlap=0.75 --set use_zones=false
mole eval outputs/hwi_parity/test.npy data/hwi-test --topk 1,5

# hwi_s22 — run.sh literally (stride 22 for codebook features and test; ~11 GPU-h)
mole codebook checkpoints/raven_checkpoint.pth data/hwi-train --out outputs/hwi_s22/train.codebook.npy \
  --kmeans-batch-size 1000000 --kmeans-n-init 1 --max-descriptors 4000000 \
  --foreground --foreground-method raven --no-invert --set window_size=224 --set overlap=0.9018 --set use_zones=false
mole embed checkpoints/raven_checkpoint.pth data/hwi-train outputs/hwi_s22/train.npy --pooling vlad \
  --codebook-from outputs/hwi_s22/train.codebook.npy --whiten-dim 384 \
  --foreground --foreground-method raven --no-vlad-intra-norm --no-invert \
  --set window_size=224 --set overlap=0.9018 --set use_zones=false
mole embed checkpoints/raven_checkpoint.pth data/hwi-test outputs/hwi_s22/test.npy --pooling vlad \
  --codebook-from outputs/hwi_s22/train.codebook.npy --whiten-from outputs/hwi_s22/train.whiten.npz \
  --foreground --foreground-method raven --no-vlad-intra-norm --no-invert \
  --set window_size=224 --set overlap=0.9018 --set use_zones=false
mole eval outputs/hwi_s22/test.npy data/hwi-test --topk 1,5

# hwi_trans — codebook AND whitening fit on the test set (the run that matches the paper)
mole codebook checkpoints/raven_checkpoint.pth data/hwi-test --out outputs/hwi_trans/test.codebook.npy \
  --kmeans-batch-size 1000000 --kmeans-n-init 1 --max-descriptors 4000000 \
  --foreground --foreground-method raven --no-invert --set window_size=224 --set overlap=0 --set use_zones=false
mole embed checkpoints/raven_checkpoint.pth data/hwi-test outputs/hwi_trans/test.npy --pooling vlad \
  --codebook-from outputs/hwi_trans/test.codebook.npy --whiten-dim 384 \
  --foreground --foreground-method raven --no-vlad-intra-norm --no-invert \
  --set window_size=224 --set overlap=0 --set use_zones=false
mole eval outputs/hwi_trans/test.npy data/hwi-test --topk 1,5
```

`hwi_w224` / `hwi_s56` are the July and 17 September runs recorded in FEATURES_RESULTS.md
(same flags as `hwi_parity` but with `--window-foreground`, mole's default k-means and all train
tokens; `hwi_s56` reuses `hwi_w224`'s codebook and whitening).

Not run: transductive fit at stride 56 (expected ≈ 0.827 from the +0.5 stride increment measured
twice on fit-on-train), and a transductive fit with the whitening only (to split the +2.7 between
codebook and PCA). Neither changes the conclusion.

## Code added along the way

- `--vlad-pooling gmp --gmp-gamma` (commit 0cd8905) and `scripts/run_gmp_ab.py`
- `--kmeans-batch-size / --kmeans-n-init` on `mole embed` and `mole codebook` (b53b8bf, 120dbfb)
- codebook pool gathered per page instead of a second full stack (3e264f3; the first parity
  attempt was OOM-killed on the 24 GB box)
