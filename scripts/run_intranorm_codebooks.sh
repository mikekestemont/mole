#!/usr/bin/env bash
# Does intra-normalization transfer to the DEPLOYABLE codebooks, and does it STACK
# with vocabulary adaptation?
#
# run_intranorm_ab.sh measured intra-norm on the .trans (per-archive refit) codebook.
# But the deployable index uses the FROZEN or ADAPTED codebook, not trans. Burstiness
# is an encoding property, not a codebook one, so it should carry over — this confirms
# it, and builds the 2x factorial (codebook x normalization) on Flanders and the rest.
#
# Reuses, from outputs/loao_adapt/:  <A>.frozen.codebook.npy, <A>.adapt.codebook.npy,
# and the PLAIN eval jsons <A>.frozen.eval.json / <A>.adapt.eval.json (default = plain).
# Only the two intra-norm embeds per archive are new.
#
#   cd ~/mole && conda activate mole
#   mkdir -p outputs/intranorm_cb
#   bash scripts/run_intranorm_codebooks.sh 2>&1 | tee outputs/intranorm_cb/run.log
set -euo pipefail

CKPT="${CKPT:-runs/pooled_bin_ft/checkpoint.pth}"
DATA="${DATA:-data}"
OUT="${OUT:-outputs/intranorm_cb}"
ADAPT_OUT="${ADAPT_OUT:-outputs/loao_adapt}"        # the frozen/adapt codebooks + plain evals
ARCHIVES="${ARCHIVES:-antwerp-bin brackley-2350 flanders-set-bin leroy-bin utrecht-bin}"
GEOM="${GEOM:---set window_size=224 --set overlap=0 --set use_zones=false}"

[ -f "$CKPT" ] || { echo "error: no checkpoint at $CKPT" >&2; exit 1; }
mkdir -p "$OUT"

intra_embed_eval () {  # $1=which(frozen|adapt)  $2=archive
  local which="$1" A="$2" cb="$ADAPT_OUT/$2.$1.codebook.npy"
  [ -f "$cb" ] || { echo "error: missing $cb (run run_loao_adapt.sh first)" >&2; exit 1; }
  mole embed "$CKPT" "$DATA/$A" "$OUT/$A.$which.intra.npy" --pooling vlad \
      --codebook-from "$cb" --vlad-intra-norm $GEOM
  mole eval "$OUT/$A.$which.intra.npy" "$DATA/$A" --topk 1,5 --cross-doc-only --per-hand \
      --out "$OUT/$A.$which.intra.eval.json"
}

for A in $ARCHIVES; do
  echo "===== $A ====="
  intra_embed_eval frozen "$A"
  intra_embed_eval adapt  "$A"
done

echo "===== intra-norm ON THE FROZEN codebook (plain frozen -> intra frozen) ====="
pairs=(); for A in $ARCHIVES; do
  pairs+=("$ADAPT_OUT/$A.frozen.eval.json" "$OUT/$A.frozen.intra.eval.json"); done
mole eval-compare "${pairs[@]}"

echo "===== intra-norm ON THE ADAPTED codebook (plain adapt -> intra adapt = the STACK) ====="
pairs=(); for A in $ARCHIVES; do
  pairs+=("$ADAPT_OUT/$A.adapt.eval.json" "$OUT/$A.adapt.intra.eval.json"); done
mole eval-compare "${pairs[@]}"

echo "===== the full deployable stack vs the naive baseline (frozen+plain -> adapt+intra) ====="
pairs=(); for A in $ARCHIVES; do
  pairs+=("$ADAPT_OUT/$A.frozen.eval.json" "$OUT/$A.adapt.intra.eval.json"); done
mole eval-compare "${pairs[@]}"
