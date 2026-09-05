#!/bin/bash
# Dependency-chained evaluation on HyperKvasir: the 2 frozen-init baselines + every
# SSL variant that has an ema_backbone.pt, then a final aggregate.
#
#   bash lrz/submit_eval_chain.sh                 # pre baselines + existing SSL ckpts
#   ONLY_PRE=1 bash lrz/submit_eval_chain.sh      # just the pre-SSL baselines
#   CORPUS=hkv_unlabeled bash lrz/submit_eval_chain.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh
mkdir -p lrz/logs

DATASET="${DATASET:-hyperkvasir}"
CORPUS="${CORPUS:-gastronet}"
DEPS=""
add() {  # $1 = label, rest = env assignments
  local label="$1"; shift
  local jid
  jid=$(env DATASET="$DATASET" "$@" sbatch --parsable lrz/sbatch_eval.sbatch)
  echo "  $label -> job $jid"
  DEPS="${DEPS:+$DEPS:}$jid"
}

echo "pre-SSL baselines ($DATASET):"
add "pre/siglip2"  INIT=siglip2  STAGE=pre
add "pre/imagenet" INIT=imagenet STAGE=pre

if [ "${ONLY_PRE:-0}" != "1" ]; then
  echo "post-SSL variants with a checkpoint (corpus=$CORPUS):"
  for OBJ in lejepa dino; do
    for INIT in siglip2 imagenet; do
      if [ -f "$OUT_ROOT/ssl/${OBJ}_${INIT}_${CORPUS}_full/ema_backbone.pt" ]; then
        add "post/${INIT}/${OBJ}/${CORPUS}" INIT="$INIT" STAGE=post OBJ="$OBJ" CORPUS="$CORPUS"
      else
        echo "  skip post/${INIT}/${OBJ}/${CORPUS} (no checkpoint yet)"
      fi
    done
  done
fi

# aggregate is pure pandas -- no GPU, runs in seconds on the login node, and
# is far more reliable there than fighting venv-visibility on compute nodes.
echo
echo "eval jobs queued: $DEPS"
echo "when they finish (squeue --me), run the report on the login node:"
echo "    source lrz/job_env.sh && python -m vlfz.report.aggregate"
echo "    cat \$OUT_ROOT/results/report.md"
