#!/usr/bin/env bash
# Does VOCABULARY ADAPTATION help, leave-one-archive-out?
#
# "All About VLAD" (Arandjelović & Zisserman, CVPR 2013) contribution #2: when a
# vocabulary is applied to data it was NOT clustered on, the fixed centres no longer
# sit at the mean of the descriptors in each cell, so residuals pick up a bias. The
# fix is cheap — keep the cells (assignment against the original centres), move each
# centre to the mean of the NEW corpus's descriptors in it. No re-clustering.
#
# This measures exactly that, as the middle of three arms per held-out archive A:
#
#   frozen   codebook fitted on the OTHER archives only, applied to A as-is
#            (A unseen in vocabulary — the deployable "add a collection" baseline)
#   adapt    that SAME frozen codebook, centres moved to A's own descriptors
#            (the paper's vocabulary adaptation — cells unchanged, centres re-estimated)
#   trans    codebook fitted on A itself (full per-archive refit — the upper bound)
#
# Headline test: adapt − frozen  (did the cheap centre move recover accuracy?).
# Context test:  adapt − trans   (how much of the full-refit gap did it close, for free?).
#
#   cd ~/GitRepos/mole && source .venv/bin/activate
#   nohup bash scripts/run_loao_adapt.sh > outputs/loao_adapt/run.log 2>&1 &
#
# Every archive dir must be BINARIZED and carry a labels.csv (mole prep does both).
# Name them <archive>-bin (etc.) so --cross-doc-only picks the right doc-id rule.
# Overridable from the environment:
#   ARCHIVES="flanders-set-bin antwerp-bin" CKPT=checkpoints/raven_checkpoint.pth \
#       bash scripts/run_loao_adapt.sh
set -euo pipefail

CKPT="${CKPT:-runs/pooled_bin_ft/checkpoint.pth}"   # pinned base backbone
DATA="${DATA:-data}"                                # per-archive dirs live here
OUT="${OUT:-outputs/loao_adapt}"
ARCHIVES="${ARCHIVES:-antwerp-bin brackley-2350 flanders-set-bin leroy-bin utrecht-bin}"
CLUSTERS="${CLUSTERS:-100}"                         # VLAD K (raven parity)
MAXDESC="${MAXDESC:-4000000}"                       # reservoir cap for fit AND adapt
MINCELL="${MINCELL:-50}"                            # adapt: keep cells below this frozen
GEOM="${GEOM:---set window_size=224 --set overlap=0 --set use_zones=false}"
PYTHON="${PYTHON:-python}"

[ -f "$CKPT" ] || { echo "error: no checkpoint at $CKPT" >&2; exit 1; }
for A in $ARCHIVES; do
  [ -d "$DATA/$A" ] || { echo "error: no archive dir $DATA/$A" >&2; exit 1; }
  [ -f "$DATA/$A/labels.csv" ] || echo "warning: $DATA/$A has no labels.csv" >&2
done
mkdir -p "$OUT"

encode_eval () {  # $1=arm  $2=archive  $3=codebook
  local arm="$1" A="$2" cb="$3"
  mole embed "$CKPT" "$DATA/$A" "$OUT/$A.$arm.npy" --pooling vlad \
      --codebook-from "$cb" $GEOM
  mole eval "$OUT/$A.$arm.npy" "$DATA/$A" --topk 1,5 --cross-doc-only --per-hand \
      --out "$OUT/$A.$arm.eval.json"
}

for A in $ARCHIVES; do
  TRAIN_DIRS=()
  for B in $ARCHIVES; do [ "$B" = "$A" ] || TRAIN_DIRS+=("$DATA/$B"); done
  echo "===== fold: $A held out  (frozen vocab fit on ${TRAIN_DIRS[*]}) ====="

  # -- arm 1: FROZEN — codebook from the other archives, A never seen ---------
  echo "-- [frozen] fitting vocabulary on the training archives"
  mole codebook "$CKPT" "${TRAIN_DIRS[@]}" --out "$OUT/$A.frozen.codebook.npy" \
      --clusters "$CLUSTERS" --max-descriptors "$MAXDESC" $GEOM
  encode_eval frozen "$A" "$OUT/$A.frozen.codebook.npy"

  # -- arm 2: ADAPT — move that frozen vocabulary's centres onto A ------------
  echo "-- [adapt]  adapting the frozen vocabulary to $A"
  mole codebook "$CKPT" "$DATA/$A" --out "$OUT/$A.adapt.codebook.npy" \
      --adapt-from "$OUT/$A.frozen.codebook.npy" --adapt-min-assigned "$MINCELL" \
      --max-descriptors "$MAXDESC" $GEOM
  encode_eval adapt "$A" "$OUT/$A.adapt.codebook.npy"

  # -- arm 3: TRANS — full per-archive refit (upper bound) -------------------
  echo "-- [trans]  refitting a codebook on $A alone"
  mole codebook "$CKPT" "$DATA/$A" --out "$OUT/$A.trans.codebook.npy" \
      --clusters "$CLUSTERS" --max-descriptors "$MAXDESC" $GEOM
  encode_eval trans "$A" "$OUT/$A.trans.codebook.npy"
done

echo "===== HEADLINE: adaptation vs frozen (does moving the centres help?) ====="
pairs=(); for A in $ARCHIVES; do pairs+=("$OUT/$A.frozen.eval.json" "$OUT/$A.adapt.eval.json"); done
mole eval-compare "${pairs[@]}"

echo "===== CONTEXT: adaptation vs full refit (how much of the gap it closes) ====="
pairs=(); for A in $ARCHIVES; do pairs+=("$OUT/$A.trans.eval.json" "$OUT/$A.adapt.eval.json"); done
mole eval-compare "${pairs[@]}"
