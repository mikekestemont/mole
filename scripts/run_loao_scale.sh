#!/usr/bin/env bash
# Does SCRIPT-SCALE NORMALIZATION help, leave-one-archive-out?
#
# For each archive: fit a VLAD codebook on ALL THE OTHERS, encode the held-out
# archive against it, and evaluate on that archive's own gallery. The held-out
# archive contributes nothing to the vocabulary — and, in the scaled arm, nothing
# to the choice of target scale either, which is the part that is easy to get
# wrong. Two arms per fold, identical except for the scale treatment:
#
#   base   codebook fitted as-is                     → no resampling anywhere
#   scale  target = pooled median of the 3 TRAIN     → fit pages AND the held-out
#          archives, fit pages resampled to it         archive resampled to it
#
# The target is measured only on the training archives, so the held-out archive
# is as unseen in scale as it is in hands. `mole embed --codebook-from` reads the
# target out of the codebook's provenance sidecar, so the two arms run the SAME
# embed command — the codebook alone decides whether pages get resampled.
#
#   cd ~/GitRepos/mole && source .venv/bin/activate
#   nohup bash scripts/run_loao_scale.sh > outputs/loao_scale/run.log 2>&1 &
#
# Every archive dir must be BINARIZED and carry a labels.csv (mole prep does both;
# see WORKFLOW.md §2b). Name them <archive>-bin so --cross-doc-only can pick the
# right doc-id rule out of mole.data.docids. Overridable from the environment:
#   ARCHIVES="leroy-bin utrecht-bin" CKPT=checkpoints/raven_checkpoint.pth \
#       bash scripts/run_loao_scale.sh
set -euo pipefail

CKPT="${CKPT:-runs/pooled_bin_ft/checkpoint.pth}"   # pinned base backbone
DATA="${DATA:-data}"                                # per-archive dirs live here
OUT="${OUT:-outputs/loao_scale}"
ARCHIVES="${ARCHIVES:-antwerp-bin brackley-2350 flanders-set-bin leroy-bin}"
CLUSTERS="${CLUSTERS:-100}"                         # VLAD K (raven parity)
SAMPLE="${SAMPLE:-200}"                             # pages/archive for the median
MAXDESC="${MAXDESC:-1000000}"                       # k-means input cap: the CPU cost
                                                    #   here rivals the ViT passes
GEOM="${GEOM:---set window_size=224 --set overlap=0 --set use_zones=false}"
PYTHON="${PYTHON:-python}"

[ -f "$CKPT" ] || { echo "error: no checkpoint at $CKPT" >&2; exit 1; }
for A in $ARCHIVES; do
  [ -d "$DATA/$A" ] || { echo "error: no archive dir $DATA/$A" >&2; exit 1; }
  [ -f "$DATA/$A/labels.csv" ] || echo "warning: $DATA/$A has no labels.csv" >&2
done
mkdir -p "$OUT"

for A in $ARCHIVES; do
  TRAIN_DIRS=()
  for B in $ARCHIVES; do [ "$B" = "$A" ] || TRAIN_DIRS+=("$DATA/$B"); done
  echo "===== fold: $A held out  (fit on ${TRAIN_DIRS[*]}) ====="

  # -- the target, from the TRAINING archives only ---------------------------
  TARGET=$("$PYTHON" -c "
import sys
from mole.prep.scale import corpus_target
t, scales = corpus_target(sys.argv[1:], sample=$SAMPLE, progress=False)
for s in scales:
    print(f'  {s.directory}: median={s.median:.1f}px spread={s.spread:.2f} '
          f'({s.n_measured} measured, {s.n_failed} failed)', file=sys.stderr)
print(f'{t:.2f}')
" "${TRAIN_DIRS[@]}")
  echo "-- pooled target from the 3 training archives: ${TARGET}px"

  # -- arm 1: baseline, no scale treatment -----------------------------------
  echo "-- [base]  fitting codebook"
  mole codebook "$CKPT" "${TRAIN_DIRS[@]}" --out "$OUT/$A.base.codebook.npy" \
      --clusters "$CLUSTERS" --max-descriptors "$MAXDESC" $GEOM
  mole embed "$CKPT" "$DATA/$A" "$OUT/$A.base.npy" --pooling vlad \
      --codebook-from "$OUT/$A.base.codebook.npy" $GEOM
  mole eval "$OUT/$A.base.npy" "$DATA/$A" --topk 1,5 --cross-doc-only --per-hand \
      --out "$OUT/$A.base.eval.json"

  # -- arm 2: everything normalized to the training target -------------------
  echo "-- [scale] fitting codebook at ${TARGET}px"
  mole codebook "$CKPT" "${TRAIN_DIRS[@]}" --out "$OUT/$A.scale.codebook.npy" \
      --clusters "$CLUSTERS" --max-descriptors "$MAXDESC" --scale-target "$TARGET" $GEOM
  # no --scale-normalize flag needed: the codebook carries the target
  mole embed "$CKPT" "$DATA/$A" "$OUT/$A.scale.npy" --pooling vlad \
      --codebook-from "$OUT/$A.scale.codebook.npy" $GEOM
  mole eval "$OUT/$A.scale.npy" "$DATA/$A" --topk 1,5 --cross-doc-only --per-hand \
      --out "$OUT/$A.scale.eval.json"
done

echo "===== the §4.2 decision rule over all folds ====="
pairs=()
for A in $ARCHIVES; do
  pairs+=("$OUT/$A.base.eval.json" "$OUT/$A.scale.eval.json")
done
mole eval-compare "${pairs[@]}"
