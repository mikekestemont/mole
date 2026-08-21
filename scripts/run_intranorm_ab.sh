#!/usr/bin/env bash
# Does INTRA-NORMALIZATION help? ("All About VLAD", CVPR 2013, contribution #1.)
#
# It's implemented (mole/embed/vlad.py) but OFF by default — the default is Raven's
# plain VLAD (residual sum → signed-sqrt power-norm → global L2). Intra-normalization
# L2-normalizes each cluster's residual block first, to stop one bursty visual word
# from dominating the page vector. This has never actually been measured on mole.
#
# One codebook per archive, TWO embeds against it — identical except the flag — so
# this isolates the normalization alone (no vocabulary difference between the arms):
#
#   plain   --no-vlad-intra-norm   (the current default; Raven parity)
#   intra   --vlad-intra-norm      (per-cluster L2 before power-norm)
#
# Reuses the transductive codebooks from run_loao_adapt.sh if they're already there
# ($OUT/<A>.trans.codebook.npy); otherwise fits one per archive.
#
#   cd ~/GitRepos/mole && source .venv/bin/activate
#   nohup bash scripts/run_intranorm_ab.sh > outputs/intranorm_ab/run.log 2>&1 &
set -euo pipefail

CKPT="${CKPT:-runs/pooled_bin_ft/checkpoint.pth}"
DATA="${DATA:-data}"
OUT="${OUT:-outputs/intranorm_ab}"
ADAPT_OUT="${ADAPT_OUT:-outputs/loao_adapt}"        # reuse its .trans codebooks if present
ARCHIVES="${ARCHIVES:-antwerp-bin brackley-2350 flanders-set-bin leroy-bin utrecht-bin}"
CLUSTERS="${CLUSTERS:-100}"
MAXDESC="${MAXDESC:-4000000}"
GEOM="${GEOM:---set window_size=224 --set overlap=0 --set use_zones=false}"

[ -f "$CKPT" ] || { echo "error: no checkpoint at $CKPT" >&2; exit 1; }
mkdir -p "$OUT"

for A in $ARCHIVES; do
  echo "===== $A ====="
  CB="$ADAPT_OUT/$A.trans.codebook.npy"
  if [ ! -f "$CB" ]; then
    CB="$OUT/$A.codebook.npy"
    echo "-- fitting a transductive codebook (no reusable one found)"
    mole codebook "$CKPT" "$DATA/$A" --out "$CB" \
        --clusters "$CLUSTERS" --max-descriptors "$MAXDESC" $GEOM
  else
    echo "-- reusing $CB"
  fi

  mole embed "$CKPT" "$DATA/$A" "$OUT/$A.plain.npy" --pooling vlad \
      --codebook-from "$CB" --no-vlad-intra-norm $GEOM
  mole eval "$OUT/$A.plain.npy" "$DATA/$A" --topk 1,5 --cross-doc-only --per-hand \
      --out "$OUT/$A.plain.eval.json"

  mole embed "$CKPT" "$DATA/$A" "$OUT/$A.intra.npy" --pooling vlad \
      --codebook-from "$CB" --vlad-intra-norm $GEOM
  mole eval "$OUT/$A.intra.npy" "$DATA/$A" --topk 1,5 --cross-doc-only --per-hand \
      --out "$OUT/$A.intra.eval.json"
done

echo "===== intra-normalization vs plain VLAD (per §4.2 decision rule) ====="
pairs=(); for A in $ARCHIVES; do pairs+=("$OUT/$A.plain.eval.json" "$OUT/$A.intra.eval.json"); done
mole eval-compare "${pairs[@]}"
