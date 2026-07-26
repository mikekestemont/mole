# MOLE — recommended workflow

The end-to-end pipeline and the exact commands, in order. This is the operational
companion to `ARCHITECTURE.md` (which covers design/decisions). Keep it current as
phases land.

Legend: ✅ available now · 🚧 coming (Phase 4+).

```
raw pages ──▶ [prep] ──▶ zones.json ──▶ [augview] (inspect)
                                    └──▶ [train] ──▶ checkpoint ──▶ [embed] ──▶ [eval]
```

---

## 0. Install (once)

Training + embedding env (CPU works for prep/augview/embed; GPU for training):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
pip install 'mole[detect]'      # YOLO text-zone detector (ultralytics + HF hub)
# On the CUDA server, install matching torch wheels — see README.
```

## 1. Put images in a dataset folder ✅

A dataset is just a folder of freely named images (`data/<name>/`). Optionally add a
partial `labels.csv` (`filename,hand_id[,confidence][,source][,notes]`) — used only
by `eval` and supervised training, never by self-supervised training.

## 2. `prep` — detect + store the text zone ✅

Runs the detector ONCE and writes `zones.json` (coordinates + detections, stamped
with model) into the dataset folder. No images are duplicated.

```bash
mole prep data/samples                       # → data/samples/zones.json (+ QC)
# options: --method yolo|heuristic  --padding 16  --conf 0.25  --sample N
#          --padding-frac 0.05  (pad by a share of the SHORT side when that is
#                                larger than --padding; for mixed-resolution corpora)
#          --write-crops DIR   (opt-in: also materialise cropped images)
#          --zones-out PATH    (default: <input_dir>/zones.json)
```

Artifacts: `data/samples/zones.json`, `outputs/prep_qc.html`.

**Padding is a coverage dial, and coverage is the metric that matters** — a zone with
extra background is nearly free (the foreground filter drops blank parchment anyway),
while a zone that clips text destroys writer signal nothing downstream can recover. On
a corpus whose pages span 500–6600 px, prefer `--padding-frac`: matched for coverage,
it over-crops the small pages far less than the equivalent absolute `--padding`.

### Fine-tuned zone detectors

`scripts/ls_to_yolo.py` + `scripts/train_zone_obb.py` produce an OBB detector emitting
a single `MainZone` class (the Label Studio class name), which is in `ZONE_FAMILIES`
alongside `Text`:

```bash
mole prep data/frags --yolo-weights runs/zones/frag-obb-v2/train/weights/best.pt \
    --padding-frac 0.05
```

### 2b. Binarize + normalize the script scale ✅

Camera photographs of manuscripts differ in two ways that have nothing to do with the
hand: **tone** (parchment colour, lighting, microfilm vs. colour) and **scale** (how many
pixels one letter is, which is set by camera distance and DPI, not by the scribe). Both
are removed here, before any windowing.

```bash
# tone: adaptive Sauvola threshold → black ink on white
mole prep data/samples --binarize sauvola --binarize-out data/samples-bin

# scale: also resample every page to a constant script module (this corpus's own median)
mole prep data/samples --binarize sauvola --binarize-out data/samples-bin \
    --normalize-scale profile
```

The **script module** is the height of the body of the writing (roughly the x-height),
measured by folding the page's row-ink profile onto one line period — a whole-page
statistic, so it does not depend on finding individual words. `--normalize-scale profile`
measures it per page and resamples the *grayscale* original before re-thresholding, so the
ink stays clean rather than being rescaled as a bitmap. `word` (word-blob heights) exists
for ablation and is considerably noisier.

This matters because a 224px window is a *physical* crop: at 12px/letter it sees a
paragraph, at 40px/letter a few characters. `--max-side` does **not** substitute for it —
that normalizes page size, and two pages of equal pixel size can still hold script at
wildly different scales. Note that normalizing may **upscale** pages past `--max-side`.

To bring several corpora into one shared scale, measure them first and pass one target:

```bash
mole scale-target data/leroy-bin data/utrecht-bin data/sluis-bin     # prints a pooled median
mole prep data/utrecht --binarize sauvola --binarize-out data/utrecht-bin \
    --normalize-scale profile --target-module 22.0
```

Artifacts: the binarized folder, plus `scale.json` inside it recording the target, the
method, and each page's module before/after. The QC sheet gains per-page module and
scale-factor columns and a header line reporting the collapse (spread before → after).

Downstream this is automatic: `mole embed` reads `scale.json` and skips re-measuring
pages `prep` already normalized. For a corpus that was *not* prepped this way, normalize
at embed time instead — and pin the target into a codebook so every later embed inherits
it:

```bash
mole codebook <ckpt> data/train outputs/cb.npy --scale-target auto   # records the target
mole embed <ckpt> data/new outputs/new.npy --codebook-from outputs/cb.npy   # applies it
mole embed <ckpt> data/new outputs/new.npy --no-scale-normalize            # or opt out
```

**Check it worked.** Re-run `mole scale-target` on the normalized folders: the per-corpus
medians should sit on the target, and the internal IQR/median spread should collapse too.
Measured on 60-page samples of three corpora, normalized to 45.4px:

| corpus | median before | after | spread before | after |
|---|---:|---:|---:|---:|
| leroy-bin | 27.8 | 45.5 | 0.31 | 0.02 |
| utrecht | 45.4 | 45.4 | 0.45 | 0.03 |
| brackley-set | 58.2 | 45.5 | 0.23 | 0.01 |

A 2.1× spread between corpora becomes 1.00×, and — the part that matters more — the
variation *within* each corpus drops by an order of magnitude, so it is genuinely
per-page, not a per-corpus constant. Note that `utrecht` was the median corpus and so
barely moves overall, yet its internal spread still falls from 0.45 to 0.03.

Then confirm retrieval did not regress, comparing like with like
(`mole eval outputs/emb.npy data/<dataset> --cross-doc-only --topk 1,5`), and skim the QC
sheet for pages with an implausible factor. The same handful of pages is reported
unmeasurable before and after (3/4/1 above) — those are failed binarizations or blank
leaves, and they are passed through untouched rather than guessed at.

## 3. Inspect / re-view the QC sheet ✅

Open `outputs/prep_qc.html` in a browser (original + detections | chosen zone | crop).

**Re-run the QC without re-detecting** (fast, no GPU — reuses `zones.json`):

```bash
mole prep data/samples --from-zones --qc outputs/prep_qc.html
```

Use this after tweaking `zones.json` by hand, or just to re-open the view. To change
the crop (e.g. more padding), re-run step 2 with a new `--padding` (re-detects), or
edit the bboxes in `zones.json` and re-run `--from-zones`.

## 4. `augview` — inspect augmentations ✅

Auto-loads `zones.json`, so windows are sampled only from inside the text zone.

```bash
mole augview data/samples --output outputs/augview.html --n-images 6 --n-views 6
# --preset mild|default|aggressive   --window-size 512   --no-zones (sample whole page)
```

Artifact: `outputs/augview.html`. Locked defaults: preset `mild`, window 512.

## 5. `train` — self-supervised pretraining ✅

Single-GPU-first (CUDA / MPS / CPU), mixed precision on CUDA. Reads `zones.json`
per dataset so windows come from the text zone. Seamless step-level resume:
Ctrl-C checkpoints cleanly; re-running auto-resumes from the run dir.

```bash
# GPU server — real training (vit_small, batch 128):
mole train configs/pretrain.yaml --output-dir runs/base_v1
mole train configs/pretrain.yaml --output-dir runs/base_v1        # auto-resumes
mole train configs/pretrain.yaml --set optim.lr=1e-4 --set train.epochs=50

# Laptop (CPU/MPS) — fast pipeline/resume sanity check only (~seconds):
mole train configs/smoke.yaml --output-dir runs/smoke

# Warm-start from an ORIGINAL AttMask/iBOT checkpoint (e.g. Raven's) or a mole one:
# loads weights only, adopts the source's architecture, starts a fresh run at step 0.
mole train configs/pretrain.yaml --output-dir runs/base_v1 --init-from /path/to/raven_checkpoint.pth
```

`--init-from` accepts a mole checkpoint, an original AttMask/iBOT run, or a bare
extracted-backbone checkpoint (`{"state_dict": …}`, what Raven ships). It rebuilds a
matching model — architecture from the checkpoint's `config`/`args`, or **inferred from
the weights** when there's no metadata — loads the backbone (heads re-initialise), and
reports what loaded vs. re-initialised. It is weight-only (not optimizer/RNG) and starts
a fresh run at step 0; re-running the same command resumes, but pointing it at a stale
dir errors rather than silently ignoring. Embed a foreign checkpoint directly too:
`mole embed /path/to/raven_checkpoint.pth data/samples outputs/raven.npy` (sidecar stamps
`source: foreign-import`) — handy to sanity-check the source model before a long run.

> If a run diverges (loss → NaN/Inf) it stops with a clear `[mole] ERROR` and guidance
> (Apple **MPS** is numerically unreliable for this — train on **CUDA**; or lower the LR;
> or start from a fresh `--output-dir`). Nothing is saved on divergence.

> The production config is heavy (vit_small on ~768 image-forwards/step) — run it on
> the GPU server, not a laptop. Use `configs/smoke.yaml` (vit_tiny, batch 16) locally.

Artifacts in the run dir: `checkpoint.pth` (rolling), `checkpoint_epochNNNN.pth`,
`manifest.json`, `config.json`, `log.txt`, and TensorBoard `events.out.tfevents.*`.
`--mode continual` (replay) lands in Phase 7; today it trains like scratch.

### Monitoring with TensorBoard

Training writes scalars into the run dir automatically: `loss/total`, `loss/cls`,
`loss/patch`, and the `sched/*` schedules (lr, weight_decay, momentum_teacher). Watch
them live — on the training machine, or locally after copying a run dir down:

```bash
tensorboard --logdir runs           # then open http://localhost:6006
```

Point `--logdir` at the parent (`runs`), not one run, so multiple runs overlay for
comparison. On the remote GPU box, tunnel the port over your VPN/SSH session:

```bash
ssh -L 6006:localhost:6006 you@gpu-server   # then run tensorboard on the server
```

Cadence is `train.tb_every_steps` (default 10); disable entirely with
`--set train.tensorboard=false`. There is **no early stopping** (per Raven's advice —
the model keeps improving); just watch `loss/total` trend down over a long run.

**Embedding projector.** Every `train.projector_every_epochs` (default 5) a snapshot of
document embeddings is logged to TensorBoard's **PROJECTOR** tab (interactive PCA / t-SNE
/ UMAP). Each snapshot is a step in the dropdown, so you can flip through them and watch
structure emerge as training proceeds; hovering shows the page thumbnail. Points carry a
`hand` / `dataset` / `file` metadata column — colour by `hand` once a `labels.csv` exists
(same-hand pages should cluster), or by `dataset` meanwhile. Tuning:
`--set train.projector_max_images=500`, `--set train.projector_every_epochs=2`, or
`--set train.projector=false` to turn it off. Works with or without labels.

## 6. `embed` — extract embeddings ✅ (Phase 5)

Loads the checkpoint's teacher ViT, samples zone-aware windows, resizes them
deterministically to `model_size`, pools, and writes `.npy` (or `.parquet`) plus a
lineage-stamped `<out>.mapping.json` sidecar. CPU/MPS/CUDA (auto).

```bash
# mean over patch tokens (default), L2-normalised page vectors
mole embed runs/base_v1/checkpoint.pth data/samples outputs/emb.npy --pooling mean

# other poolings; vlad saves a reproducible <out>.codebook.npy bound to the model id
mole embed <ckpt> data/samples outputs/emb.npy --pooling cls
mole embed <ckpt> data/samples outputs/vlad.npy --pooling vlad --vlad-clusters 64 --seed 0
mole embed <ckpt> data/samples outputs/patches.npy --pooling patches   # raw per-patch rows

# optional PCA-whitening; force device; override embed geometry
mole embed <ckpt> data/samples outputs/emb.npy --whiten --device cpu --set window_size=384
```

## 7. `eval` ✅ / `models` 🚧 (Phase 6)

```bash
# Writer-retrieval benchmark (leave-one-out): mAP, macro-mAP, Top-k, and a
# cross-dataset breakdown when labels span >1 dataset. Writes <emb>.eval.json.
mole eval outputs/emb.npy data/<dataset>            # cosine ranking (default)
mole eval outputs/emb.npy data/<dataset> --metric euclidean --topk 1,5,10

mole models list                                    # lineage tree (still 🚧)
```

Labels come from each dataset's `labels.csv` (`filename,hand_id`); only labeled
images are scored, matched by dataset folder + basename. Relevance = same
`hand_id`. Historical-WI reproduction: arrange the test set as a dataset folder
with `hand_id` = writer ID and embed with the Raven-parity VLAD flags
(`--pooling vlad --vlad-clusters 100 --foreground --no-vlad-intra-norm`).
The binarized HWI images are white-on-black, so train with `--set data.invert=true`
(recorded in the checkpoint; `mole embed` inherits it — or force with `--invert`)
to put them in the black-on-white regime the foreground filter assumes.

VLAD codebook: by default it is fit on the set being embedded (transductive). To
fit on the **training** split and apply to test (Raven's protocol), embed the
train set first (that saves `<out>.codebook.npy`), then pass it to the test embed
with `--codebook-from <train>.codebook.npy`.

---

## Artifact locations (all git-ignored)

| Artifact | Path |
|---|---|
| datasets (+ `zones.json`, `labels.csv`) | `data/<name>/` |
| prep QC sheet, augview grid | `outputs/` |
| optional materialised crops | wherever `--write-crops` points |
| training runs / checkpoints | `runs/` |
| models registry | `models/` |
