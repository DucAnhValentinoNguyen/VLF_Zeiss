#!/bin/bash
#SBATCH --job-name=vlfz_ssl
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=lrz/logs/%x_%j.out
#SBATCH --error=lrz/logs/%x_%j.err
#
# Self-resubmitting SSL pretraining for ONE variant.
#   OBJ=lejepa INIT=siglip2 STAGE=full sbatch lrz/sbatch_ssl_pretrain.sbatch
# Resubmits itself until ckpt/DONE exists.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source lrz/job_env.sh
wait_for_gpu || { echo "resubmitting (bad node)"; sbatch "$0"; exit 0; }

OBJ="${OBJ:-lejepa}"; INIT="${INIT:-siglip2}"; STAGE="${STAGE:-full}"
CORPUS="${CORPUS:-gastronet}"
OUT_DIR="$OUT_ROOT/ssl/${OBJ}_${INIT}_${CORPUS}_${STAGE}"
mkdir -p "$OUT_DIR"

echo "[$(date)] SSL $OBJ/$INIT/$CORPUS/$STAGE  job=$SLURM_JOB_ID  -> $OUT_DIR"
srun python -m vlfz.ssl.pretrain \
  --objective "$OBJ" --init "$INIT" --stage "$STAGE" --corpus "$CORPUS" --resume \
  ${SSL_EXTRA:-}

if [ -f "$OUT_DIR/DONE" ]; then
  echo "[$(date)] DONE -> $OUT_DIR/ema_backbone.pt"
else
  echo "[$(date)] not finished -> resubmitting"
  sbatch "$0"
fi
