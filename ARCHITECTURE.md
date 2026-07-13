# MOLE — Architecture & Build Status

Living design + progress doc for the `mole` rewrite. Read this first when resuming
a session. It records what is **locked**, what is **parked**, and how to continue.

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
selfsup/     AttMask pretraining, continual updates, finetuning   [stubs → Phase 4]
supervised/  triplet/metric + probes (SCAFFOLD ONLY, interfaces fixed early)
embed/       pooling (mean/cls/vlad/patches), reproducible VLAD, extraction  [stubs → Phase 5]
lineage/     model registry, versioning, provenance  [stubs → Phase 6]
eval/        retrieval benchmark from partial labels  [stubs → Phase 6]
cli/         typer entry points (one per command)
config.py    YAML load + --set overrides (schema deferred to Phase 4)
```

CLI ≙ Python API (feature parity). Stubbed commands print a "Phase N" notice and
exit 1, so `mole --help` renders fully from day one. Heavy imports (torch/kornia)
are lazy inside functions so `import mole` / `mole --help` stay fast.

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
| 4 `mole train` (port + resume + RNG state) | ⬜ next candidate |
| 5 `mole embed` (mean/cls/patches; VLAD w/ fixed seed) | ⬜ |
| 6 Lineage registry + `mole models` + eval | ⬜ |
| 7 Continual + finetune (replay shards, LoRA vs full = open) | ⬜ |
| 8 Supervised scaffold → implementation | ⬜ |
| 9 README + config docs | ⬜ |

**Data is already text-cropped**, so Phase 3 is optional for current material and
Phase 4 (training) can proceed without it.

---

## Parked / open decisions

- **Phase 3 text-zone detector (BUILT).** `mole prep` in `src/mole/prep/`:
  `detect.py` (pluggable `TextZoneDetector`: `HeuristicTextZoneDetector` ink-density
  CV default-safe backend + `YoloTextZoneDetector`), `run.py` (`prep_folder` → crops +
  records), `qc.py` (self-contained HTML contact sheet: original+overlays vs crop,
  fallbacks flagged). CLI `mole prep IN OUT --method yolo|heuristic --padding --conf
  --sample --qc`.
  - **YOLO backend = `magistermilitum/YOLO_manuscripts`** ("YOLO-gen", Sergio Torres
    Aguilar) — YOLOv11x-OBB, **MIT**, plain `ultralytics` (opt-in `mole[detect]`),
    trained on e-NDP (Parisian registers 1326–1504) + CATMuS + HORAE. Main zone =
    union of class FAMILIES `Text` (Text/Text_Main) + `Initial` (drop capitals &
    decorated/historiated initials — they carry scribe signal); excludes `Paratext`
    (marginalia), Decoration, Marks, Damage. Family match = label.split('_')[0], via
    `ZONE_FAMILIES` in detect.py. OBB → boxes follow skew (deskew angle free later).
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
- VLAD k-means had **no seed** (not reproducible) → Phase 5 fixes with fixed seed +
  saved, model-versioned codebook.
- Resume was epoch-granular, **no RNG state**, no Ctrl-C handler, hard dist dependency,
  `.cuda()` hardcoded → Phase 4 adds seamless resume + single-GPU-first.
- `extract_embeddings.py` reimplements its own ViT (divergence risk) → Phase 5 imports
  the canonical model.
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
