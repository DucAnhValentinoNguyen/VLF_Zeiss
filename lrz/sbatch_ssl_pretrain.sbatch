#!/bin/bash
#SBATCH --job-name=vlfz_ssl
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=06:00:00   # 3 epochs @ bs128 ~= LeJEPA <1h / DINO ~2h on a healthy node; the resubmit trap below covers a slow node / TIMEOUT
#SBATCH --signal=B:TERM@180   # SIGTERM 3 min before the wall so the EXIT trap can resubmit before SIGKILL
#SBATCH --exclude=lrz-hgx-h100-003   # degraded 2026-09-09: ran LeJEPA at 0.23 it/s (~11x slow) -> both jobs there TIMEOUT'd. Remove once fixed.
#SBATCH --output=lrz/logs/%x_%j.out
#SBATCH --error=lrz/logs/%x_%j.err
#
# Self-resubmitting SSL pretraining for ONE variant.
#   OBJ=lejepa INIT=siglip2 STAGE=full sbatch lrz/sbatch_ssl_pretrain.sbatch
# Resumes from last.ckpt and resubmits itself until $OUT_DIR/DONE exists -- via an
# EXIT trap, so a SLURM TIMEOUT / preemption (which kills the python line before
# any trailing shell code) still triggers the resubmit, not only a clean exit.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source lrz/job_env.sh

OBJ="${OBJ:-lejepa}"; INIT="${INIT:-siglip2}"; STAGE="${STAGE:-full}"
CORPUS="${CORPUS:-gastronet}"
OUT_DIR="$OUT_ROOT/ssl/${OBJ}_${INIT}_${CORPUS}_${STAGE}"
[ -n "${RUN_TAG:-}" ] && OUT_DIR="${OUT_DIR}__${RUN_TAG}"
mkdir -p "$OUT_DIR"

HANDLED=0
finish() {
  rc=$?
  [ "$HANDLED" = 1 ] && return 0
  if [ -f "$OUT_DIR/DONE" ]; then
    echo "[$(date)] DONE -> $OUT_DIR/ema_backbone.pt"
  elif [ -f "$OUT_DIR/last.ckpt" ]; then
    echo "[$(date)] exited rc=$rc without DONE, last.ckpt present -> resubmit (resume)"
    sbatch "$0"
  else
    echo "[$(date)] exited rc=$rc with no checkpoint -> NOT resubmitting (investigate the log)"
  fi
}
trap finish EXIT

wait_for_gpu || { echo "bad node -> resubmit"; HANDLED=1; sbatch "$0"; exit 0; }

echo "[$(date)] SSL $OBJ/$INIT/$CORPUS/$STAGE  job=$SLURM_JOB_ID  node=$SLURMD_NODENAME  -> $OUT_DIR"
# no srun: single-node single-GPU, and srun does not reliably carry the sbatch
# env (SSL_EPOCHS / GASTRONET_SOURCE / venv PATH) onto the step on this cluster.
"$VIRTUAL_ENV/bin/python" -m vlfz.ssl.pretrain \
  --objective "$OBJ" --init "$INIT" --stage "$STAGE" --corpus "$CORPUS" --resume \
  ${SSL_EXTRA:-}
