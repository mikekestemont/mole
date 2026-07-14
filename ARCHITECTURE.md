# MOLE — Architecture & Build Status

Living design + progress doc for the `mole` rewrite. Read this first when resuming
a session. It records what is **locked**, what is **parked**, and how to continue.
For the operational command-by-command pipeline, see `WORKFLOW.md`.

MOLE = continual self-supervised handwriting embeddings for premodern documents,
refactored from Tim Raven's adaptation of **AttMask** (Kakogeorgiou et al.),
itself in the **DINO** / **iBOT** lineage. Named after the Mexican sauce remade
from yesterday's leftovers — the model is continually re-pretrained on old + new
data.

---

## Working protocol (governs every session)

1. **Do not run training / GPU jobs / heavy tests.** The user has a CUDA server and
   runs/tests everything. Agent tool use = read + write files, at most syntax-level
   checks (`py_compile`) and cheap CPU visualizations. Token economy matters.
2. **Phased delivery.** Stop after each phase, summarize, list decisions, wait for
   sign-off before continuing.
3. **Critical design decisions → present 2–3 options + a recommendation, let the
   user choose.** Never silently decide.
4. Give the user exact test commands + what to check; design phases testable in
   minutes.
5. Small reviewable diffs; no speculative refactors beyond the brief.

The user runs a local (macOS, Python 3.10, CPU/MPS) venv for cheap checks and a
CUDA server for training.

---

## Package layout (`src/mole/`, sklearn-style paradigm separation)

```
data/        loaders, dataset manifests + partial labels, patch-windows, augmentations, augview
prep/        (PARKED) main-text-zone isolation + QC contact sheet
selfsup/     AttMask pretraining (vit/head/attmask/wrapper/loss/dataset/train/checkpoint) [Phase 4 DONE]
supervised/  triplet/metric + probes (SCAFFOLD ONLY, interfaces fixed early)
embed/       pooling (mean/cls/vlad/patches), reproducible VLAD, extraction  [Phase 5 DONE]
lineage/     model registry, versioning, provenance  [stubs → Phase 6]
eval/        retrieval benchmark from partial labels  [stubs → Phase 6]
cli/         typer entry points (one per command)
config.py    YAML load + --set overrides + config_hash (schema: data/aug/model/mask/optim/loss/train)
progress.py  track(iterable) + progress_bar(total) — the loop-progress helpers
```

Configs: `configs/pretrain.yaml` (GPU, vit_small/batch128) and `configs/smoke.yaml`
(laptop sanity, vit_tiny/batch16, ~seconds). `mole train` is single-GPU-first
(CUDA/MPS/CPU), AMP on CUDA, seamless STEP-LEVEL resume (all RNG states, Ctrl-C
ckpt, auto-resume from the run dir, "already complete" guard), dual aligned
progress bars. Two smoke-fixed bugs recorded in git log: iBOT-center buffer shape
drift (relaxed buffer load) and RNG-state CPU/uint8 coercion on GPU resume.

CLI ≙ Python API (feature parity). Stubbed commands print a "Phase N" notice and
exit 1, so `mole --help` renders fully from day one. Heavy imports (torch/kornia)
are lazy inside functions so `import mole` / `mole --help` stay fast.

**Progress convention (project-wide):** every long-running loop (prep, augview,
training, embedding, k-means, eval) wraps its iterable in `mole.progress.track(...)`
(tqdm-based). No bare loops for anything >~1–2 s; no direct `tqdm` calls — route
through `track` so the look is consistent and the backend is swappable in one place.

---

## Locked design decisions

| Decision | Value | Rationale |
|---|---|---|
| Python floor | **3.10** | user's server is 3.10.13; code is 3.10-safe (`from __future__ import annotations`) |
| Resolution: `window_size` | **512 px** | physical crop from page; 256 (original) was too zoomed for writer style on wide charters — 512 ≈ 4–6 words / 3–4 lines |
| Resolution: `model_size` | **224 px** | standard ViT input; window→224 for BOTH train and embed (embed no longer feeds 256 raw + pos-embed interpolation) |
| patch overlap | 0.5 | 512@0.5 yields ~24–63 windows/page on samples; raise to ~0.7 in Phase 4 if sample-starved |
| Augmentation preset | **`mild`** | chosen visually via `mole augview`; enough invariance without smearing detail. `TRAINING_PRESET` in `data/augment.py` |
| Color | **invariant** | random grayscale + jitter → generalizes to unseen materials (color/microfilm/bitonal mix) |
| Rotation | ±4°, off in `mild` | scan-skew robustness; white fill (≈parchment). On in default/aggressive only |
| Flips | none, ever | per brief (orientation is signal) |
| Normalize | ToTensor only (no ImageNet mean/std) | matches the ported checkpoint's training domain; left as a future config switch |
| Env split | training vs `prep` in separate envs | kraken (if ever used) pins conflicting torch; `requirements-prep.txt` is separate |

**Augmentation presets** (`data/augment.py::PRESETS`): `mild` / `default` /
`aggressive`. `augview` samples a **random window per view** from across the page
(true training draw = random location + augmentation), same window sequence reused
across presets for fair comparison.

---

## Phase status

| Phase | State |
|---|---|
| 0 Audit | ✅ done |
| 1 Skeleton + env | ✅ done (`import mole`, `mole --help` verified) |
| 2 Data + augmentations + `mole augview` | ✅ done (preset + window_size locked visually) |
| 3 `mole prep` (text-zone detector) | ✅ done — heuristic + YOLO backends, QC contact sheet |
| 4 `mole train` (port + resume + RNG state) | ✅ done — single-GPU-first, step-level resume, Ctrl-C ckpt |
| 5 `mole embed` (mean/cls/patches; VLAD w/ fixed seed) | ✅ done — teacher-ViT load, deterministic resize, 4 poolings, reproducible VLAD, lineage-stamped sidecar |
| 6 Lineage registry + `mole models` + eval | ⬜ **← NEXT** |
| 7 Continual + finetune (replay shards, LoRA vs full = open) | ⬜ |
| 8 Supervised scaffold → implementation | ⬜ |
| 9 README + config docs | ⬜ |

**Data is already text-cropped**, so Phase 3 is optional for current material and
Phase 4 (training) can proceed without it.

### Phase 5 `mole embed` — BUILT (how it works)

`src/mole/embed/`: `extract.py` (driver), `pooling.py` (`Pooling` enum +
`pool_window`/`patch_descriptors`), `vlad.py` (reproducible codebook + encode).
- `load_backbone(ckpt)` loads the checkpoint's **teacher** into the canonical
  `mole.selfsup.vit` (never a re-implemented ViT). `_teacher_backbone_state` strips
  the MultiCropWrapper `backbone.` prefix and drops the iBOT `head.`/`fc.` Identities;
  `masked_embed` is tolerated-if-absent. `strict=False` load, hard-errors on any real
  missing weight. Returns `(model, meta)`; `meta.model_id = f"{arch}@{config_hash[:8]}+step{N}"`.
- `_page_index` mirrors `PatchWindowDataset` (auto-discovers `zones.json`, windows
  from image sizes only). `_build_transform` = deterministic `Resize((model_size),BICUBIC)`
  + `ToTensor` (no ImageNet normalise — matches training's tensor contract; NO random
  crop, NO raw-256 pos-embed interpolation — Phase-2 decision).
- Pooling: `mean` (default, patch-token mean per window → mean over windows, L2'd),
  `cls` (class token(s) flattened), `patches` (raw per-patch descriptors, one output
  row per descriptor), `vlad`. VLAD fits ONE seeded k-means codebook over all page
  descriptors (`sklearn` if present, else a seeded numpy k-means++ fallback — both
  reproducible, verified bit-identical across runs), encodes per page, saves the
  codebook as `<out>.codebook.npy`.
- Output `.npy` (or `.parquet` via pandas) + sidecar `<out>.mapping.json` stamped with
  `model_id`/`embed_dim`/rows. `_warn_on_version_mismatch` warns if the output dir
  already holds a sidecar from a different model. Optional `--whiten` (PCA, transductive).
- CLI wired: `mole embed <ckpt> <in> <out> --pooling mean|cls|patches|vlad [--whiten]
  [--batch-size N] [--vlad-clusters K] [--seed S] [--device ...] [--set window_size=..]`.
- **Verified** on CPU with `runs/smoke` (vit_tiny) + `runs/base_v1` (vit_small): all
  four poolings produce correct shapes, VLAD reproducible, version warning fires.

### Foreign-checkpoint warm-start (interop with Raven's original checkpoints)

`mole.selfsup.checkpoint` normalises both mole and original AttMask/iBOT checkpoints:
`normalize_checkpoint` detects a foreign file (`args` Namespace + `module.`-prefixed
DDP student, vs mole's `config` + `global_step`), strips the `module.` prefix, and
synthesizes a mole `config` — architecture (`ARCH_FIELDS`) recovered from `args`, the
rest left at mole defaults (deliberately NOT the original's rejected 256 px window).
`filtered_load` does a strict-safe load (only matching-shape keys; reports missing /
shape-mismatch / unexpected so a warm-start never crashes on a partial fit).
- `mole train <config> --init-from raven.pth`: fresh run at **step 0**, weights only
  (no optimizer/RNG), adopts the source arch (overrides conflicting config leaves with
  a warning), loads student+teacher (student seeded from teacher if the file is
  teacher-only). Ignored when resuming. This is a warm-start, NOT Phase-7 finetune
  (no no-mutate branch / replay / LoRA).
- `mole embed raven.pth ...`: `load_backbone` runs the same normaliser, so an original
  checkpoint embeds directly; sidecar `source: foreign-import`. ViT is a faithful port
  (identical param names, `img_size=[224]` both sides → `pos_embed` shapes match), so
  weights load cleanly when arch/patch_size/num_class_tokens agree.
- **Verified** on CPU/MPS with a synthesized Raven-format checkpoint (DDP `module.`
  student, `args` not `config`): warm-start loaded all params, embed produced
  `foreign-import` output, resume-precedence + shape-mismatch + teacher-only paths all
  exercised.

### Resume here (next session) — Phase 6 lineage registry + `mole models` + eval

Stubs: `src/mole/lineage/registry.py`, `src/mole/eval/retrieval.py`; CLI `mole models
list/show` and `mole eval` currently print the Phase-6 stub notice. The embed sidecar
already stamps `model_id`/`config_hash`/`global_step` — the registry should build on
that. Eval consumes `labels.csv` (partial labels, `mole.data.datasets.load_labels`)
against an embeddings `.npy` + its `.mapping.json`; metrics = mAP / top-k / cross-dataset.

---

## Parked / open decisions

- **Phase 3 text-zone detector (BUILT).** `mole prep` in `src/mole/prep/`:
  `detect.py` (pluggable `TextZoneDetector`: `HeuristicTextZoneDetector` ink-density
  CV default-safe backend + `YoloTextZoneDetector`), `run.py` (`prep_folder`),
  `qc.py` (self-contained HTML contact sheet: original+overlays vs crop, fallbacks
  flagged). CLI `mole prep IN --method yolo|heuristic --padding --conf --sample --qc
  [--write-crops DIR] [--zones-out PATH]`.
  - **Preprocessing reuse = zone manifest, not re-cropped images (LOCKED design).**
    prep runs the detector ONCE and writes `zones.json` into the dataset folder
    (`src/mole/data/zones.py`: `ZoneManifest`/`ZoneEntry`, stamped with detector +
    model + padding + families + per-detection boxes). Auto-discovered downstream like
    `labels.csv`. `sample_windows(..., bounds=bbox)` restricts patch-windows to the
    zone; augview auto-loads it (`--zones PATH` / `--no-zones`). Physical crops are
    opt-in (`--write-crops`). Rationale: no image duplication, no re-running YOLO per
    training iteration, padding/classes re-derivable from stored detections, and
    IIIF-streaming-compatible (store coords, not pixels). This also fixed augview
    sampling windows from page background/clutter on full pages.
  - **YOLO backend = `magistermilitum/YOLO_manuscripts`** ("YOLO-gen", Sergio Torres
    Aguilar) — YOLOv11x-OBB, **MIT**, plain `ultralytics` (opt-in `mole[detect]`),
    trained on e-NDP (Parisian registers 1326–1504) + CATMuS + HORAE. Main zone =
    union of class family `Text` (Text/Text_Main) only; Initial/Paratext/Decoration/
    Marks/Damage excluded (tried including Initial, reverted — main text alone cropped
    better). Family match = label.split('_')[0], via `ZONE_FAMILIES` in detect.py.
    OBB → boxes follow skew (deskew angle free later).
    https://huggingface.co/magistermilitum/YOLO_manuscripts
  - **Verified** on the 11 sample charters: loads + detects Text/Text_Main on every
    page (+ Initial on the decorated one), 16 s total incl. 118 MB download, CPU. Samples
    are pre-cropped so zone ≈ whole strip — real value is on full pages. Weights `best.pt`
    (118 MB) fetched once via `hf_hub_download`, cached.
  - **Rejected:** kraken; YALTAi/PonteIneptique (Clérice, @polyneptique) — Kraken
    *adapter* (custom code, buggy) not a clean weights drop; its Segmonto dataset
    (HF `biglam/yalta_ai_segmonto_manuscript_dataset`, Zenodo 6814770) kept for optional
    future fine-tuning. Bill Mattingly = Qwen VL *transcription*, not detection.
  - **Roadmap:** heuristic/YOLO auto-labels → fine-tune a tiny in-domain YOLO if needed.
- **Finetune method (Phase 7):** full finetune vs LoRA/adapter — present options then.
- **Config schema (Phase 4):** field set intentionally not frozen yet.

## Open questions still owed by the user (bind at Phase 4+)

- Hardware/scale: single vs multi-GPU, VRAM, images per dataset / total (→ DDP,
  streaming, VLAD feasibility).
- Labeled data volume: how many hands/images, and do any hands span multiple
  datasets/digitizations (cross-dataset pairs = the valuable confound detector).

---

## Audit findings that shape later phases (from Phase 0)

- Original code = DINO/iBOT + AttMask. Color was thrown away (`.convert('L')`) despite
  a 3-channel model → fixed by color-invariant augmentation on a 3-channel model.
- Two inconsistent patch schemes (train 256→224 vs embed 256 raw) → unified on
  window→224.
- VLAD k-means had **no seed** (not reproducible) → Phase 5 FIXED: fixed seed +
  saved, model-versioned codebook (`embed/vlad.py`; verified bit-identical reruns).
- Resume was epoch-granular, **no RNG state**, no Ctrl-C handler, hard dist dependency,
  `.cuda()` hardcoded → Phase 4 adds seamless resume + single-GPU-first.
- `extract_embeddings.py` reimplements its own ViT (divergence risk) → Phase 5 FIXED:
  `embed/extract.py` loads teacher weights into the canonical `mole.selfsup.vit`.
- Competition-specific to delete: `icdar2017patcher` naming, `subset_of_Imagenet_train_split`,
  `combine_ckpt.py`, hardcoded `/data/traven/...` paths, dead commented mask code in
  `loader.py`, the unused `ibot` aug (contains forbidden horizontal flip).

Source being ported lives in `attmask/` (kept in-repo as reference).

---

## How to resume / test commands

```bash
# local cheap env (CLI + augview, CPU)
python3 -m venv .venv && . .venv/bin/activate
pip install -e . --no-deps && pip install "typer>=0.12" "rich>=13"
pip install torch torchvision kornia numpy pillow      # for augview

mole --help
mole augview data/samples --output outputs/augview.html   # window 512, all presets
```

Test images: `data/samples/` (git-ignored). Currently 11 RGB charters (1319–1354),
already text-cropped, no DPI metadata.
