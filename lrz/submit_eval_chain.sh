#!/bin/bash
# Dependency-chained evaluation: the 2 frozen-init baselines + the 4 SSL variants,
# then a final aggregate. Uses --dependency=afterany so one bad node doesn't stall
# the chain.
#
#   bash lrz/submit_eval_chain.sh              # pre baselines + whatever SSL ckpts exist
#   ONLY_PRE=1 bash lrz/submit_eval_chain.sh   # just the pre-SSL baselines
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh
mkdir -p lrz/logs

DEPS=""
add() {  # $1 = human label, rest = env assignments
  local label="$1"; shift
  local jid
  jid=$(env "$@" sbatch --parsable lrz/sbatch_eval.sbatch)
  echo "  $label -> job $jid"
  DEPS="${DEPS:+$DEPS:}$jid"
}

echo "pre-SSL baselines:"
add "pre/siglip2"  INIT=siglip2  STAGE=pre
add "pre/imagenet" INIT=imagenet STAGE=pre

if [ "${ONLY_PRE:-0}" != "1" ]; then
  echo "post-SSL variants (only those with an ema_backbone.pt):"
  for OBJ in lejepa dino; do
    for INIT in siglip2 imagenet; do
      if [ -f "$OUT_ROOT/ssl/${OBJ}_${INIT}_full/ema_backbone.pt" ]; then
        add "post/${INIT}/${OBJ}" INIT="$INIT" STAGE=post OBJ="$OBJ"
      else
        echo "  skip post/${INIT}/${OBJ} (no checkpoint yet)"
      fi
    done
  done
fi

echo "final aggregate (after: $DEPS)"
sbatch --dependency=afterany:"$DEPS" lrz/sbatch_aggregate.sbatch
echo "done. squeue --me"
