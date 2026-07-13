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
mole train configs/pretrain.yaml --output-dir runs/base_v1
mole train configs/pretrain.yaml --output-dir runs/base_v1        # auto-resumes
mole train configs/pretrain.yaml --set optim.lr=1e-4 --set train.epochs=50
# quick smoke: --set model.arch=vit_tiny --set optim.batch_size=32 --set data.num_workers=0
```

Artifacts in the run dir: `checkpoint.pth` (rolling), `checkpoint_epochNNNN.pth`,
`manifest.json`, `config.json`, `log.txt`. `--mode continual` (replay) lands in
Phase 7; today it trains like scratch.

## 6. `embed` — extract embeddings 🚧 (Phase 5)

```bash
mole embed runs/base_v1/checkpoint.pth data/samples outputs/emb.npy --pooling mean
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
