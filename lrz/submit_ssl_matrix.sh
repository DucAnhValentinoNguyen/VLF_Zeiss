#!/bin/bash
# Queue the SSL pretraining matrix. Single-GPU, self-resubmitting.
#
#   bash lrz/submit_ssl_matrix.sh              # 2 runs: LeJEPA+DINO x imagenet (SigLIP-2 dropped, 2026-09-16)
#   TIER=lejepa bash lrz/submit_ssl_matrix.sh   # LeJEPA only (1 run) — cheapest
#   INITS="siglip2 imagenet" bash lrz/submit_ssl_matrix.sh   # dormant: re-add siglip2 if ever needed
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh
mkdir -p lrz/logs

STAGE="${STAGE:-full}"
CORPUS="${CORPUS:-gastronet}"    # gastronet | hkv_unlabeled
TIER="${TIER:-all}"             # all | lejepa | dino
case "$TIER" in
  lejepa) OBJS=(lejepa) ;;
  dino)   OBJS=(dino) ;;
  *)      OBJS=(lejepa dino) ;;
esac
read -ra INITS <<< "${INITS:-imagenet}"   # imagenet only by default; SigLIP-2 dropped from the matrix

for OBJ in "${OBJS[@]}"; do
  for INIT in "${INITS[@]}"; do
    echo "submit SSL $OBJ/$INIT/$CORPUS/$STAGE"
    OBJ="$OBJ" INIT="$INIT" STAGE="$STAGE" CORPUS="$CORPUS" sbatch lrz/sbatch_ssl_pretrain.sbatch
  done
done
echo "queued. watch: squeue --me ; tail -f lrz/logs/vlfz_ssl_*.out"
