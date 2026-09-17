#!/usr/bin/env python
"""Write copies of binarized archives resampled to ONE script module — for training.

`mole embed --codebook-from <cb>` rescales every page to the codebook's pinned
module on the fly, so an index built that way already lives at one scale.
Training has no such hook: `mole train` reads pages as they sit on disk. Before a
pooled finetune over archives prepped at different scales, materialise what the
embed step does, with the same :class:`PageScaler`, so the model trains at
exactly the scale the index uses.

Bitonal pages are resampled as bitmaps (LANCZOS down / BICUBIC up, no
re-threshold) — the same operation the Phase A embeddings went through. Cleaner
would be grayscale source → resample → Sauvola (`mole prep --normalize-scale
profile --target-module PX`), when the sources are at hand.

Each output folder gets the pages under their ORIGINAL names (labels.csv and
doc_ids.csv are copied; zones.json is not — its coordinates would be stale) and
a scale.json recording target + per-page factor, so a later `mole embed` sees
a residual factor of ~1 and touches nothing.

Usage (server, from the repo root):
    python scripts/materialize_scale.py --target 41.16 --suffix=-s41 \\
        data/antwerp-bin data/utrecht-charters data/leroy-sauvola data/comital
    # -> data/antwerp-bin-s41, data/utrecht-charters-s41, ...
    # the target is the one the Phase A codebook pinned:
    #   python -c "import json;print(json.load(open('outputs/cross/fit.codebook.npy.json'))['script_module_target'])"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from mole.prep.scale import (PageScaler, ScaleEntry, ScaleManifest, find_scale, load_scale,
                             save_scale, scale_meta)
from mole.progress import track

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CARRIED = ("labels.csv", "doc_ids.csv")


def materialize(src: Path, dst: Path, target: float, *, method: str = "profile") -> dict:
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} exists and is not empty — refusing to overwrite")
    dst.mkdir(parents=True, exist_ok=True)
    found = find_scale(src)
    manifest = load_scale(found) if found else None
    scaler = PageScaler(target, method=method, manifest=manifest)
    out_manifest = ScaleManifest()
    files = sorted(p for p in src.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    for p in track(files, f"Resampling {src.name}", unit="page"):
        page = Image.open(p)
        page.load()
        res = scaler.rescale(page, name=p.name)
        out = dst / f"{p.stem}.png"
        (res.image.convert("L") if res.image.mode != "L" else res.image).save(out)
        out_manifest.images[out.name] = ScaleEntry(
            module=res.module, scale=res.factor, size=tuple(res.image.size),
            module_out=(res.module * res.factor) if res.module else None)
    for name in CARRIED:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    summary = {k: v for k, v in scaler.summary().items()
               if k not in ("script_module_target", "method")}
    out_manifest.meta = scale_meta(
        target, method, f"materialize_scale from {src}", **summary)
    save_scale(dst / "scale.json", out_manifest)
    return scaler.summary()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path, help="Binarized archive folders.")
    ap.add_argument("--target", type=float, required=True, help="Script module in px.")
    ap.add_argument("--suffix", default="-s41",
                    help="Appended to each folder name (write --suffix=-s41: it starts with a dash).")
    ap.add_argument("--method", default="profile", help="profile | word (mole.prep.scale).")
    args = ap.parse_args()
    for src in args.dirs:
        dst = src.parent / f"{src.name}{args.suffix}"
        s = materialize(src, dst, args.target, method=args.method)
        print(f"  {src.name} -> {dst.name}: {s['rescaled']}/{s['pages']} resampled, "
              f"median ×{s['median_scale']} (module {s['median_module']} -> {args.target:g}px), "
              f"{s['from_manifest']} from scale.json, {s['unmeasurable']} unmeasurable")


if __name__ == "__main__":
    main()
