#!/usr/bin/env python
"""Convert a Label Studio JSON-MIN export of `MainZone` polygons into a YOLO dataset.

This is the bridge from your in-browser corrections to a fine-tunable detector.
You annotate each page's text zone(s) as polygons in Label Studio (single class,
`MainZone`), export **JSON-MIN**, and this writes a YOLO dataset — images plus
per-image label files plus `data.yaml` — ready for an ultralytics fine-tune.

Two output geometries, because your fragments sit skewed on mounts:

* ``--obb`` (default): each polygon becomes an **oriented** box via the min-area
  rotated rectangle (``cv2.minAreaRect``), format ``cls x1 y1 x2 y2 x3 y3 x4 y4``
  (normalised). This hugs a tilted charter without scooping mount into the corners
  — the whole reason we chose OBB — and matches the magistermilitum OBB weights.
* ``--aabb``: axis-aligned ``cls cx cy w h`` (normalised), the format
  ``train_zone_detector.py`` already consumes.

KEY POINT ON COORDINATES. Label Studio stores polygon points as **percentages**,
so they are resolution-independent: we ignore the (downscaled, for-display)
size in the export and re-derive pixel coordinates from the **full-resolution**
original on disk (``--images``). Annotation precision is therefore not lost to the
1400px display downscale.

NEGATIVES ARE KEPT. A page you left with no polygon (a ruler-only shot, an
effaced scrap) is written as an image with an **empty** label file. In detection
training an unlabelled region is an implicit negative, so these teach the model to
*reject* rulers/scales/mounts — do not drop them.

    python scripts/ls_to_yolo.py --export project-min.json \
        --images ~/GitRepos/scripy/data/harvest/md-hires \
        --out runs/zones/frag-obb --obb

Then fine-tune with scripts/train_zone_obb.py (OBB) or train_zone_detector.py (AABB).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CONTROL_NAME = "label"  # matches PolygonLabels name="label" in ls_config.xml


def task_image_path(task: dict, by_name: dict) -> Path | None:
    """Resolve a task's original image.

    The generator stamps `filename` into each task. It may be an **absolute path**
    (multi-collection batches, where basenames can collide across folders) or a bare
    basename (single-folder batches). Prefer the absolute path when it exists;
    otherwise fall back to a basename lookup in ``--images``."""
    fn = task.get("filename")
    if not fn:
        return None
    p = Path(fn)
    if p.is_absolute() and p.exists():
        return p
    return by_name.get(os.path.basename(fn))


def task_regions(task: dict) -> list[dict]:
    """Polygon regions for a task under the control name (missing/empty = negative)."""
    regs = task.get(CONTROL_NAME)
    return regs if isinstance(regs, list) else []


def points_to_pixels(points_pct, w: int, h: int):
    """LS percentage points (0–100) → pixel coords at the true image size."""
    return [(p[0] / 100.0 * w, p[1] / 100.0 * h) for p in points_pct]


def rect_to_pixels(r: dict, w: int, h: int):
    """A Label Studio rectangle region → its 4 corner points in pixels.

    LS rectangles are ``x, y`` (top-left, %), ``width, height`` (%), and
    ``rotation`` (degrees, clockwise **around the top-left corner**). We rotate
    the local corners and scale to the true image size, so an oriented box drawn
    on a skewed fragment converts straight to an OBB.
    """
    import math

    x, y = r["x"], r["y"]
    ww, hh = r["width"], r["height"]
    a = math.radians(r.get("rotation") or 0.0)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for lx, ly in ((0, 0), (ww, 0), (ww, hh), (0, hh)):
        rx = x + lx * ca - ly * sa
        ry = y + lx * sa + ly * ca
        pts.append((rx / 100.0 * w, ry / 100.0 * h))
    return pts


def region_pixels(r: dict, w: int, h: int):
    """Corner/vertex points (px) for a region, whether rectangle or polygon."""
    if "points" in r:                              # PolygonLabels
        return points_to_pixels(r["points"], w, h)
    if "width" in r and "height" in r:             # RectangleLabels (± rotation)
        return rect_to_pixels(r, w, h)
    return []


def obb_corners(points_px):
    """Min-area rotated rectangle around a polygon → 4 corner points."""
    import cv2
    import numpy as np

    pts = np.asarray(points_px, dtype=np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))  # (4, 2)
    return [(float(x), float(y)) for x, y in box]


def aabb(points_px):
    xs = [p[0] for p in points_px]
    ys = [p[1] for p in points_px]
    return min(xs), min(ys), max(xs), max(ys)


def label_lines(regions, w: int, h: int, *, obb: bool) -> list[str]:
    """YOLO label lines for one image (class 0 = MainZone throughout)."""
    lines = []
    for r in regions:
        pts = region_pixels(r, w, h)
        if len(pts) < 3:
            continue
        if obb:
            corners = obb_corners(pts)
            coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in corners)
            lines.append(f"0 {coords}")
        else:
            x0, y0, x1, y1 = aabb(pts)
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def build(export: Path, images: Path, out: Path, *, obb: bool, val_frac: float, seed: int):
    from PIL import Image

    tasks = json.loads(Path(export).read_text())
    by_name = ({p.name: p for p in images.iterdir() if p.suffix.lower() in IMAGE_EXT}
               if images else {})

    items = []          # (image_path, [label_lines])
    n_pos = n_neg = n_missing = 0
    for t in tasks:
        img = task_image_path(t, by_name)
        if img is None:
            n_missing += 1
            continue
        with Image.open(img) as im:
            w, h = im.size
        lines = label_lines(task_regions(t), w, h, obb=obb)
        items.append((img, lines))
        if lines:
            n_pos += 1
        else:
            n_neg += 1

    if not items:
        raise SystemExit(f"no tasks matched images in {images} "
                         f"(missing filenames: {n_missing})")

    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, round(len(items) * val_frac))
    splits = {"val": items[:n_val], "train": items[n_val:]}

    for split, rows in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lines in rows:
            link = out / "images" / split / img.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(img.resolve())
            (out / "labels" / split / f"{img.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: MainZone\n")

    kind = "OBB" if obb else "axis-aligned"
    print(f"[ls_to_yolo] {kind} dataset → {out}")
    print(f"  {len(items)} pages  ({n_pos} with zones, {n_neg} negatives)  "
          f"| {len(splits['train'])} train / {len(splits['val'])} val"
          + (f"  | {n_missing} tasks had no matching image" if n_missing else ""))


def _self_test():
    """Validate the geometry maths against the Label Studio percentage convention."""
    # A rectangle from (10%,20%) to (60%,80%) on a 1000x500 image → px (100,100)-(600,400).
    reg = [{"points": [[10, 20], [60, 20], [60, 80], [10, 80]], "polygonlabels": ["MainZone"]}]
    aabb_line = label_lines(reg, 1000, 500, obb=False)[0].split()
    cx, cy, bw, bh = map(float, aabb_line[1:])
    assert abs(cx - 0.35) < 1e-6 and abs(cy - 0.5) < 1e-6, aabb_line
    assert abs(bw - 0.5) < 1e-6 and abs(bh - 0.6) < 1e-6, aabb_line
    obb_line = label_lines(reg, 1000, 500, obb=True)[0].split()
    assert obb_line[0] == "0" and len(obb_line) == 9, obb_line
    xs = [float(obb_line[i]) for i in (1, 3, 5, 7)]
    ys = [float(obb_line[i]) for i in (2, 4, 6, 8)]
    assert abs(min(xs) - 0.10) < 1e-3 and abs(max(xs) - 0.60) < 1e-3, obb_line
    assert abs(min(ys) - 0.20) < 1e-3 and abs(max(ys) - 0.80) < 1e-3, obb_line
    # Same box as an (unrotated) RectangleLabels region → identical AABB.
    rect = [{"x": 10, "y": 20, "width": 50, "height": 60, "rotation": 0,
             "rectanglelabels": ["MainZone"]}]
    r_line = label_lines(rect, 1000, 500, obb=False)[0].split()
    rcx, rcy, rbw, rbh = map(float, r_line[1:])
    assert abs(rcx - 0.35) < 1e-6 and abs(rcy - 0.5) < 1e-6, r_line
    assert abs(rbw - 0.5) < 1e-6 and abs(rbh - 0.6) < 1e-6, r_line
    # A 90° rotation about the top-left swaps the footprint's width/height.
    rot = [{"x": 50, "y": 20, "width": 40, "height": 10, "rotation": 90,
            "rectanglelabels": ["MainZone"]}]
    pts = rect_to_pixels(rot[0], 1000, 500)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert abs((max(xs) - min(xs)) - 100.0) < 1e-3, pts   # 10% of 1000 → 100px wide
    assert abs((max(ys) - min(ys)) - 200.0) < 1e-3, pts   # 40% of 500 → 200px tall
    # Empty regions → no lines (kept as a negative image downstream).
    assert label_lines([], 100, 100, obb=True) == []
    print("[ls_to_yolo] self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, help="Label Studio JSON-MIN export file.")
    ap.add_argument("--images", type=Path, default=None,
                    help="Folder of full-resolution originals. Optional: only needed "
                         "when tasks carry bare basenames; multi-collection batches "
                         "carry absolute paths and resolve without it.")
    ap.add_argument("--out", type=Path, default=Path("runs/zones/frag-obb"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--obb", action="store_true", help="Oriented boxes (default).")
    g.add_argument("--aabb", action="store_true", help="Axis-aligned boxes.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true", help="Run geometry checks and exit.")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.export:
        ap.error("--export is required (or use --self-test)")
    build(args.export, args.images, args.out, obb=not args.aabb,
          val_frac=args.val_frac, seed=args.seed)


if __name__ == "__main__":
    main()
