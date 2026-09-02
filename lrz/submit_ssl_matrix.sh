#!/bin/bash
# Queue the SSL pretraining matrix. Single-GPU, self-resubmitting.
#
#   bash lrz/submit_ssl_matrix.sh            # Tier A: 4 runs (LeJEPA+DINO x siglip2+imagenet)
#   TIER=lejepa bash lrz/submit_ssl_matrix.sh   # LeJEPA only (2 runs) — cheapest
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh
mkdir -p lrz/logs

STAGE="${STAGE:-full}"
TIER="${TIER:-all}"     # all | lejepa | dino
case "$TIER" in
  lejepa) OBJS=(lejepa) ;;
  dino)   OBJS=(dino) ;;
  *)      OBJS=(lejepa dino) ;;
esac

for OBJ in "${OBJS[@]}"; do
  for INIT in siglip2 imagenet; do
    echo "submit SSL $OBJ/$INIT/$STAGE"
    OBJ="$OBJ" INIT="$INIT" STAGE="$STAGE" sbatch lrz/sbatch_ssl_pretrain.sbatch
  done
done
echo "queued. watch: squeue --me ; tail -f lrz/logs/vlfz_ssl_*.out"
