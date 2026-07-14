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
#          --write-crops DIR   (opt-in: also materialise cropped images)
#          --zones-out PATH    (default: <input_dir>/zones.json)
```

Artifacts: `data/samples/zones.json`, `outputs/prep_qc.html`.

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

`--init-from` reads the source's `args` to rebuild a matching model (so the weights
load), strips the DDP `module.` prefix, and reports what loaded vs. re-initialised. It
is weight-only (not optimizer/RNG) and is ignored when the run is resuming. You can
also embed a foreign checkpoint directly: `mole embed /path/to/raven_checkpoint.pth
data/samples outputs/raven.npy` (sidecar stamps `source: foreign-import`) — handy to
sanity-check the source model before a long run.

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

## 7. `eval` / `models` 🚧 (Phase 6)

```bash
mole eval outputs/emb.npy data/samples        # mAP / top-k from partial labels
mole models list                              # lineage tree
```

---

## Artifact locations (all git-ignored)

| Artifact | Path |
|---|---|
| datasets (+ `zones.json`, `labels.csv`) | `data/<name>/` |
| prep QC sheet, augview grid | `outputs/` |
| optional materialised crops | wherever `--write-crops` points |
| training runs / checkpoints | `runs/` |
| models registry | `models/` |
