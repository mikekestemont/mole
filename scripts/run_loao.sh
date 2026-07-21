#!/usr/bin/env bash
# Leave-one-archive-out evaluation of the Tier-1 supervised head — end to end.
#
# For each archive: train a head on ALL THE OTHERS (that archive contributes
# nothing to training or model selection), embed it with that head, and compare
# against its existing baseline embedding. Every hand in the archive is therefore
# an unseen class, so no --holdout-hands restriction is needed and the numbers are
# directly comparable to the archive's usual full-gallery macro-mAP.
#
# The heads are seconds of CPU each and all share ONE feature cache; the GPU cost
# is exactly one `mole embed` pass per archive.
#
#   bash scripts/run_loao.sh                      # defaults below
#   nohup bash scripts/run_loao.sh > outputs/sup_loao/run.log 2>&1 &   # detached
#
# Everything is overridable from the environment, e.g.
#   ARCHIVES="flanders-set-bin utrecht-bin" bash scripts/run_loao.sh
set -euo pipefail

CKPT="${CKPT:-runs/pooled_bin_ft/checkpoint.pth}"   # pinned base backbone
POOL="${POOL:-data/pooled-bin}"                     # labels root (all archives)
DATA="${DATA:-data}"                                # per-archive dirs live here
CACHE="${CACHE:-runs/sup_head_v1/cache}"            # the ONE GPU pass, reused
CONFIG="${CONFIG:-configs/sup_head.yaml}"
RUNS="${RUNS:-runs/sup_loao}"                       # one head per fold
OUT="${OUT:-outputs/sup_loao}"
BASE="${BASE:-outputs/pooled_final}"                # existing baseline embeddings
ARCHIVES="${ARCHIVES:-antwerp-bin brackley-2350 flanders-set-bin leroy-bin utrecht-bin}"
POOLING="${POOLING:-vlad}"                          # must match how BASE was pooled
TRAIN="${TRAIN:-1}"                                 # 0 = reuse the heads in $RUNS
#
# The head is TRAINED on window descriptors (the mean of a window's foreground
# tokens), so `POOLING=mean` reads out the very statistic the loss optimised,
# while vlad deliberately discards the mean in favour of residuals to K
# centroids. Comparing the two isolates aggregation from supervision — but only
# if BASE was pooled the same way as the candidate:
#   POOLING=vlad BASE=outputs/pooled_final ...     (deployed configuration)
#   POOLING=mean BASE=outputs/mean_pool    ...     (matched to training)

if [ ! -f "$CACHE/cache.npy" ]; then
  echo "error: no feature cache at $CACHE" >&2
  echo "  build it once with: mole sup cache $CKPT $POOL $CACHE --labeled-only" >&2
  exit 1
fi

mkdir -p "$RUNS" "$OUT"

if [ "$TRAIN" = "1" ]; then
  echo "== 1/3  training one head per fold (CPU, shared cache) =="
  for A in $ARCHIVES; do
    echo "-- fold: $A held out"
    mole sup train "$CONFIG" "$CKPT" "$POOL" --out "$RUNS/$A" \
        --holdout-archive "$A" --cache "$CACHE"
  done
else
  echo "== 1/3  SKIPPED (TRAIN=0) — reusing the heads in $RUNS =="
fi

echo "== 2/3  embedding each archive with the head that never saw it (GPU) =="
for A in $ARCHIVES; do
  echo "-- embed: $A  (pooling=$POOLING)"
  mole embed "$CKPT" "$DATA/$A" "$OUT/$A.head.npy" --head "$RUNS/$A/head.pt" \
      --pooling "$POOLING" \
      --set window_size=224 --set overlap=0 --set use_zones=false
  mole eval "$OUT/$A.head.npy" "$DATA/$A" --topk 1,5 --cross-doc-only --per-hand \
      --out "$OUT/$A.head.eval.json"
  # baseline: the pinned backbone WITHOUT the head, same --cross-doc-only rule
  mole eval "$BASE/$A.npy" "$DATA/$A" --topk 1,5 --cross-doc-only --per-hand \
      --out "$OUT/$A.base.eval.json"
done

echo "== 3/3  the §4.2 decision rule over all folds =="
pairs=()
for A in $ARCHIVES; do
  pairs+=("$OUT/$A.base.eval.json" "$OUT/$A.head.eval.json")
done
mole eval-compare "${pairs[@]}"
