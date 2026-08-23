#!/usr/bin/env bash
# Does cropping to the DETECTED text zone (fine-tuned v3 detector) improve RETRIEVAL?
#
# This is the payoff the zone detector was built for. GT layout cropping was worth
# +0.053 macro on Antwerp (measured), but that used ground-truth zones on one archive;
# no retrieval run has ever used the detector's zones. This measures it, per archive.
#
# For each archive: detect zones with the fine-tuned detector (writes zones.json),
# then embed + eval TWICE, identical except for use_zones:
#
#   off   whole page          (use_zones=false — the current default everywhere)
#   on    windows inside the   (use_zones=true — windows restricted to the detected
#         detected text zone    text-zone bbox, +--padding-frac slack)
#
# Each arm fits its OWN transductive VLAD codebook: cropping changes the descriptor
# pool, so the codebook is refit per arm (as in the Antwerp GT finding). The only
# variable between the two arms is the zone restriction.
#
#   cd ~/mole && conda activate mole      # detector inference needs mole[detect]
#   mkdir -p outputs/zones_ab
#   bash scripts/run_zones_ab.sh 2>&1 | tee outputs/zones_ab/run.log
#
# ⚠️ COVERAGE FIRST: clipping text is fatal for writer signal. Skim each archive's
# prep QC (outputs/zones_ab/<A>.qc.html) before trusting the numbers — a zone that
# under-covers will show as a retrieval DROP that is a detector bug, not a real result.
set -euo pipefail

CKPT="${CKPT:-runs/pooled_bin_ft/checkpoint.pth}"
WEIGHTS="${WEIGHTS:-runs/zones/frag-obb-v3/train/weights/best.pt}"
DATA="${DATA:-data}"
OUT="${OUT:-outputs/zones_ab}"
ARCHIVES="${ARCHIVES:-antwerp-bin brackley-2350 flanders-set-bin leroy-bin utrecht-bin}"
PADFRAC="${PADFRAC:-0.05}"
GEOM="${GEOM:---set window_size=224 --set overlap=0}"     # use_zones is the variable, set below

[ -f "$CKPT" ]    || { echo "error: no checkpoint at $CKPT" >&2; exit 1; }
[ -f "$WEIGHTS" ] || { echo "error: no detector weights at $WEIGHTS" >&2; exit 1; }
mkdir -p "$OUT"

embed_eval () {  # $1=arm(off|on)  $2=archive  $3=use_zones(true|false)
  local arm="$1" A="$2" uz="$3"
  mole embed "$CKPT" "$DATA/$A" "$OUT/$A.$arm.npy" --pooling vlad $GEOM --set "use_zones=$uz"
  mole eval "$OUT/$A.$arm.npy" "$DATA/$A" --topk 1,5 --cross-doc-only --per-hand \
      --out "$OUT/$A.$arm.eval.json"
}

for A in $ARCHIVES; do
  echo "===== $A ====="
  echo "-- detecting zones with the fine-tuned detector (writes $DATA/$A/zones.json)"
  mole prep "$DATA/$A" --method yolo --yolo-weights "$WEIGHTS" --padding-frac "$PADFRAC" \
      --qc "$OUT/$A.qc.html"
  echo "-- [off] whole page"
  embed_eval off "$A" false
  echo "-- [on]  windows inside the detected zone"
  embed_eval on "$A" true
done

echo "===== zones ON vs OFF — does detector cropping help retrieval? (§4.2 rule) ====="
pairs=(); for A in $ARCHIVES; do pairs+=("$OUT/$A.off.eval.json" "$OUT/$A.on.eval.json"); done
mole eval-compare "${pairs[@]}"
