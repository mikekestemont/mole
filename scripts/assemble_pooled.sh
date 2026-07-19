#!/usr/bin/env bash
# Assemble a pooled multi-archive dataset directory out of SYMLINKS — no copies.
#
# Each argument is an existing *binarized* archive directory (flat images + labels.csv).
# Each becomes a subfolder of the pool, which is exactly the layout
# PatchWindowDataset expects (it walks root + its immediate subdirs, one level deep).
# One shared `invert: true` in the config covers them all — every archive is Sauvola
# black-on-white on disk, fed white-on-black to raven.
#
# Re-runnable and EXTENSIBLE: when the next archive is binarized, just re-run with it
# appended to the list — `ln -sfn` refreshes existing links and adds the new one.
#
# Usage (on the GPU box, from the repo root):
#   bash scripts/assemble_pooled.sh \
#       data/antwerp-bin data/brackley-2350 data/utrecht-bin data/flanders-set-bin
#   # later, when archive #5 lands:
#   bash scripts/assemble_pooled.sh \
#       data/antwerp-bin data/brackley-2350 data/utrecht-bin data/flanders-set-bin data/<archive5>-bin
#
# Override the destination with POOL=... (default data/pooled-bin).
set -euo pipefail

POOL="${POOL:-data/pooled-bin}"

if [ "$#" -lt 2 ]; then
  echo "error: pass at least two archive directories to pool." >&2
  echo "usage: POOL=data/pooled-bin bash scripts/assemble_pooled.sh <archive-dir> <archive-dir> [...]" >&2
  exit 2
fi

mkdir -p "$POOL"
for src in "$@"; do
  if [ ! -d "$src" ]; then
    echo "error: not a directory: $src" >&2
    exit 1
  fi
  name="$(basename "$src")"
  # absolute target so the link resolves regardless of where the pool sits
  abs="$(cd "$src" && pwd -P)"
  ln -sfn "$abs" "$POOL/$name"
  n=$(find -L "$POOL/$name" -maxdepth 1 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' \) \
        | wc -l | tr -d ' ')
  printf '  linked %-28s -> %s  (%s images)\n' "$name" "$abs" "$n"
done

echo "pooled dir ready: $POOL"
echo "point training at it with:  --set data.path=$POOL   (config default already is)"
