#!/usr/bin/env python
"""Fine-tune an in-domain text-zone detector from PAGE XML ground truth.

The off-the-shelf detector (`magistermilitum/YOLO_manuscripts`) has been the
weak link since July — visibly wrong on Utrecht even on un-binarized originals —
and zones were dropped from every binarized run because of it. This trains a
replacement on layout ground truth you already own, and, just as importantly,
**measures whether the replacement is actually better** on held-out pages.

    python scripts/train_zone_detector.py \\
        --images ~/Desktop/images/antw_img_orig \\
        --page   ~/Desktop/images/page \\
        --out    runs/zones/antwerp --epochs 60

Then use it anywhere `mole prep` runs a detector:

    mole prep data/utrecht --method yolo --yolo-weights runs/zones/antwerp/weights/best.pt

WHAT IS OPTIMISED, AND WHY NOT mAP. The costs here are asymmetric: a zone that
keeps extra background is nearly free, because the contrast foreground filter
discards blank parchment anyway, while a zone that clips text destroys writer
signal nothing downstream can recover. So the headline is **text coverage**
(fraction of the true text box retained), with tightness reported beside it.
Detection mAP would average those two failure modes into a single number and
hide which one happened. The script also sweeps padding, since the cheapest
insurance against clipping is simply to grow the box.

A BASELINE IS ALWAYS REPORTED. The fine-tuned model is scored against the
off-the-shelf detector on the same held-out pages, because "we trained a model"
is not a result — "it beats what we already had" is.

WHY AXIS-ALIGNED AND NOT OBB, MEASURED. The Antwerp crops were made by
POLYGON-masking the page (everything outside the quadrilateral set to white)
before cropping to the bbox, so an axis-aligned detector cannot reproduce the
masking. That gap is negligible on this data: over 471 ground-truth regions the
bbox over-covers the polygon by a median of **1.021x** (90th pct 1.074, max
1.199), because these are flatbed scans and the quads are near-axis-aligned. OBB
would buy ~2% of area and cost a task conversion, so `detect` it is.

That is also why the base weights are COCO yolo11n rather than
magistermilitum/YOLO_manuscripts: the manuscript model is an OBB net, a
different ultralytics task with a different head, and it cannot be fine-tuned on
a detect-format dataset. On a single large high-contrast region per page, generic
pretraining is plenty; --base yolo11s.pt is the knob if more capacity is wanted,
though nano may well be the better size at 400 images.
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def build_dataset(images: Path, page: Path, out: Path, *, val_frac: float,
                  seed: int) -> tuple[Path, int, int]:
    """Write a YOLO dataset (symlinked images + normalised boxes) and data.yaml."""
    from mole.prep.pagexml import read_page_dir

    layouts = read_page_dir(page)
    by_stem = {p.stem: p for p in images.iterdir() if p.suffix.lower() in IMAGE_EXT}

    pairs = []
    for stem, layout in sorted(layouts.items()):
        img = by_stem.get(stem)
        bbox = layout.text_bbox()
        if img is None or bbox is None or not (layout.width and layout.height):
            continue
        pairs.append((img, bbox, layout.width, layout.height))
    if not pairs:
        raise SystemExit(f"no image/XML pairs found between {images} and {page}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * val_frac)))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for split, items in splits.items():
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, (x0, y0, x1, y1), w, h in items:
            link = img_dir / img.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(img.resolve())          # symlink: no 0.9 GB copy
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            (lbl_dir / f"{img.stem}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    yaml = out / "data.yaml"
    yaml.write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: Text\n")   # match ZONE_FAMILIES casing by convention
    print(f"[mole] dataset: {len(splits['train'])} train / {len(splits['val'])} val "
          f"→ {out}")
    return yaml, len(splits["train"]), len(splits["val"])


def val_split(images: Path, page: Path, *, val_frac: float, seed: int):
    """The held-out pairs, reconstructed exactly as build_dataset shuffled them.

    Same filter, same sorted order, same seeded shuffle — so scoring a checkpoint
    later never silently scores it on pages it trained on.
    """
    from mole.prep.pagexml import read_page_dir

    layouts = read_page_dir(page)
    by_stem = {p.stem: p for p in images.iterdir() if p.suffix.lower() in IMAGE_EXT}
    pairs = []
    for stem, layout in sorted(layouts.items()):
        img = by_stem.get(stem)
        bbox = layout.text_bbox()
        if img is None or bbox is None or not (layout.width and layout.height):
            continue
        pairs.append((img, bbox, layout.width, layout.height))
    random.Random(seed).shuffle(pairs)
    return pairs[:max(1, int(round(len(pairs) * val_frac)))]


def _predict_bbox(detector, image_path):
    from mole.prep.detect import main_text_zone, union_bbox

    dets = detector.detect(image_path)
    zone = main_text_zone(dets)
    if zone is not None:
        return zone
    return union_bbox(dets)


def score(detector, val_items, paddings) -> dict:
    """Coverage / tightness / IoU on held-out pages, at several padding values."""
    from mole.prep.detect import box_iou, excess_area, pad_bbox, text_coverage

    rows = {p: {"cov": [], "exc": [], "iou": []} for p in paddings}
    misses = 0
    for img, truth, w, h in val_items:
        pred = _predict_bbox(detector, img)
        if pred is None:
            misses += 1
            continue
        for p in paddings:
            b = pad_bbox(pred, p, w, h)
            rows[p]["cov"].append(text_coverage(b, truth))
            rows[p]["exc"].append(excess_area(b, truth))
            rows[p]["iou"].append(box_iou(b, truth))
    out = {"n": len(val_items), "no_detection": misses, "by_padding": {}}
    for p, r in rows.items():
        if not r["cov"]:
            continue
        out["by_padding"][p] = {
            "coverage_median": statistics.median(r["cov"]),
            "coverage_min": min(r["cov"]),
            "clipped_frac": sum(1 for c in r["cov"] if c < 0.99) / len(r["cov"]),
            "excess_median": statistics.median(r["exc"]),
            "iou_median": statistics.median(r["iou"]),
        }
    return out


def _print_score(name: str, s: dict) -> None:
    print(f"\n  {name}   ({s['n']} pages, {s['no_detection']} with no detection)")
    print(f"    {'pad':>5} {'coverage':>9} {'worst':>7} {'clipped':>8} "
          f"{'excess':>7} {'IoU':>6}")
    for p, r in sorted(s["by_padding"].items()):
        print(f"    {p:>5} {r['coverage_median']:>9.4f} {r['coverage_min']:>7.3f} "
              f"{r['clipped_frac']:>7.1%} {r['excess_median']:>7.2f} "
              f"{r['iou_median']:>6.3f}")


def _print_guidance() -> None:
    print("\n  Read COVERAGE first: clipping text is unrecoverable, extra background "
          "is cheap.\n  Pick the smallest padding whose 'clipped' column is ~0%, then "
          "check 'excess'.\n  Layout cropping was measured worth +0.053 macro on "
          "Antwerp (GT zones), so the\n  bar is whether this detector recovers a "
          "useful share of that on unseen pages.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--page", type=Path, required=True, help="Folder of PAGE XMLs.")
    ap.add_argument("--out", type=Path, default=Path("runs/zones/detector"))
    ap.add_argument("--base", default="yolo11n.pt",
                    help="Starting weights. yolo11n/s (COCO) is plenty for one big "
                         "region per page; a manuscript-pretrained .pt also works.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="Pages are ~3500px; 1024 keeps the region edges crisp.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--padding", type=int, nargs="+", default=[0, 16, 48, 96])
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip scoring the off-the-shelf detector for comparison.")
    ap.add_argument("--dataset-only", action="store_true",
                    help="Build the YOLO dataset and stop (no training).")
    ap.add_argument("--eval-only", type=Path, default=None, metavar="WEIGHTS",
                    help="Skip training and score an existing .pt on the held-out "
                         "split — e.g. a run's current best.pt while it is still "
                         "training, or after stopping it early. Ultralytics picks "
                         "best.pt by a mAP-blend fitness that weighs over- and "
                         "under-cropping symmetrically; this scores what we "
                         "actually care about.")
    args = ap.parse_args()

    val_items = val_split(args.images, args.page, val_frac=args.val_frac,
                          seed=args.seed)

    if args.eval_only:
        from mole.prep.detect import YoloTextZoneDetector

        print(f"\n{'=' * 68}\n== Held-out zone quality ({len(val_items)} pages "
              f"never trained on)\n{'=' * 68}")
        _print_score(f"{args.eval_only}",
                     score(YoloTextZoneDetector(weights=str(args.eval_only),
                                                device=args.device),
                           val_items, args.padding))
        if not args.no_baseline:
            _print_score("off-the-shelf YOLO_manuscripts",
                         score(YoloTextZoneDetector(device=args.device),
                               val_items, args.padding))
        _print_guidance()
        return

    args.out.mkdir(parents=True, exist_ok=True)
    yaml, n_train, n_val = build_dataset(args.images, args.page, args.out,
                                         val_frac=args.val_frac, seed=args.seed)
    if args.dataset_only:
        return

    from ultralytics import YOLO

    print(f"[mole] training {args.base} for {args.epochs} epochs at imgsz={args.imgsz}")
    model = YOLO(args.base)
    model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, seed=args.seed, device=args.device,
                project=str(args.out), name="train", exist_ok=True)
    best = args.out / "train" / "weights" / "best.pt"
    print(f"[mole] ✓ weights → {best}")

    from mole.prep.detect import YoloTextZoneDetector

    print(f"\n{'=' * 68}\n== Held-out zone quality ({n_val} pages never trained on)\n"
          f"{'=' * 68}")
    tuned = YoloTextZoneDetector(weights=str(best), device=args.device)
    _print_score("fine-tuned (this corpus)", score(tuned, val_items, args.padding))

    if not args.no_baseline:
        base = YoloTextZoneDetector(device=args.device)      # off-the-shelf
        _print_score("off-the-shelf YOLO_manuscripts", score(base, val_items,
                                                             args.padding))

    _print_guidance()


if __name__ == "__main__":
    main()
