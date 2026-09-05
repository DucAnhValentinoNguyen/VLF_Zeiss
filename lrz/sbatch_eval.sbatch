#!/bin/bash
#SBATCH --job-name=vlfz_eval
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=lrz/logs/%x_%j.out
#SBATCH --error=lrz/logs/%x_%j.err
#
# Evaluate ONE (dataset, init, stage[, objective, corpus]) on all classification
# tasks, plus the segmentation task when the dataset has one.
#   DATASET=hyperkvasir INIT=siglip2 STAGE=pre sbatch lrz/sbatch_eval.sbatch
#   INIT=siglip2 STAGE=post OBJ=lejepa CORPUS=gastronet sbatch lrz/sbatch_eval.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source lrz/job_env.sh
wait_for_gpu || { echo "resubmitting (bad node)"; sbatch "$0"; exit 0; }

DATASET="${DATASET:-hyperkvasir}"
INIT="${INIT:-siglip2}"; STAGE="${STAGE:-pre}"; OBJ="${OBJ:-lejepa}"
CORPUS="${CORPUS:-gastronet}"
PROTOCOLS="${PROTOCOLS:-knn}"

ARGS=(--dataset "$DATASET" --init "$INIT" --stage "$STAGE" --objective "$OBJ"
      --corpus "$CORPUS" --task all --protocols "$PROTOCOLS")

echo "[$(date)] eval ${DATASET}/${INIT}/${STAGE}/${OBJ}/${CORPUS}  job=$SLURM_JOB_ID"
srun python -m vlfz.eval.run_eval "${ARGS[@]}"
# segmentation (HyperKvasir only)
if [ "$DATASET" = "hyperkvasir" ]; then
  srun python -m vlfz.eval.seg --init "$INIT" --stage "$STAGE" --objective "$OBJ" --corpus "$CORPUS" \
    || echo "seg step skipped"
fi
