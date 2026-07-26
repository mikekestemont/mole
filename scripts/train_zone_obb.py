#!/usr/bin/env python
"""Fine-tune an ORIENTED (OBB) text-zone detector from the manuscript-pretrained weights.

The companion to ls_to_yolo.py. It consumes the YOLO-OBB dataset that script
writes from your Label Studio corrections and fine-tunes the `magistermilitum/
YOLO_manuscripts` OBB model on your Fragmentarium zones — i.e. it *adapts* a model
that already knows medieval script to your mounted-fragment imaging condition,
which is the whole diagnosis (domain shift, not task difficulty).

    # 1. build the dataset from your export
    python scripts/ls_to_yolo.py --export project-min.json \
        --images ~/GitRepos/scripy/data/harvest/md-hires --out runs/zones/frag-obb --obb
    # 2. fine-tune
    python scripts/train_zone_obb.py --data runs/zones/frag-obb/data.yaml \
        --out runs/zones/frag-obb --epochs 100

Why fine-tune the OBB weights rather than start from COCO: unlike the single big
axis-aligned region per flatbed page that train_zone_detector.py handles, mounted
fragments are skewed and the manuscript model already carries the right script
prior — so adaptation converges on far fewer pages (dozens, not hundreds).

Requires the `detect` extra's ultralytics; run it in an env that has torch + a GPU
if you have one (`--device 0`), or CPU for a small set.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def default_base() -> str:
    """The manuscript-pretrained OBB weights, fetched/cached from the HF hub."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download("magistermilitum/YOLO_manuscripts", "best.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="data.yaml from ls_to_yolo.py --obb")
    ap.add_argument("--out", type=Path, default=Path("runs/zones/frag-obb"))
    ap.add_argument("--base", default=None,
                    help="Starting weights (default: magistermilitum/YOLO_manuscripts OBB).")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Ceiling, not a target: `best.pt` is the best-VAL checkpoint, "
                         "and --patience stops early at a plateau, so a high ceiling "
                         "costs nothing but a safety margin.")
    ap.add_argument("--patience", type=int, default=20,
                    help="Early-stop after N epochs without val improvement. Off a "
                         "pretrained model on a small set, ~20 usually stops well "
                         "before the ceiling. Set 0 to disable.")
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="Fragments run large; 1024 keeps zone edges crisp.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="e.g. 0 for CUDA, mps, or cpu.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    base = args.base or default_base()
    print(f"[train_zone_obb] fine-tuning {base}\n  data={args.data}  epochs={args.epochs}")
    model = YOLO(base)
    if model.task != "obb":
        raise SystemExit(
            f"base weights are task={model.task!r}, not 'obb' — pass an OBB model, "
            f"or build the dataset with ls_to_yolo.py --aabb and use train_zone_detector.py")
    # project MUST be absolute — ultralytics resolves a relative one under its own
    # runs dir (the bug train_zone_detector.py already documents).
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, seed=args.seed, device=args.device,
                patience=args.patience, project=str(args.out.resolve()),
                name="train", exist_ok=True)
    best = Path(getattr(model.trainer, "save_dir", args.out.resolve() / "train")) / "weights" / "best.pt"
    print(f"[train_zone_obb] ✓ weights → {best}")
    print("  Use in mole:  mole prep <dir> --method yolo --yolo-weights " + str(best))


if __name__ == "__main__":
    main()
