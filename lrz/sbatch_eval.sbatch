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
# Evaluate ONE (init, stage[, objective]) on both REAL-Colon tasks.
#   INIT=siglip2 STAGE=pre sbatch lrz/sbatch_eval.sbatch
#   INIT=siglip2 STAGE=post OBJ=lejepa sbatch lrz/sbatch_eval.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source lrz/job_env.sh
wait_for_gpu || { echo "resubmitting (bad node)"; sbatch "$0"; exit 0; }

INIT="${INIT:-siglip2}"; STAGE="${STAGE:-pre}"; OBJ="${OBJ:-lejepa}"
PROTOCOLS="${PROTOCOLS:-knn,eknn}"
EDL_HEAD="${EDL_HEAD:-1}"

ARGS=(--init "$INIT" --stage "$STAGE" --objective "$OBJ" --task all --protocols "$PROTOCOLS")
[ "$EDL_HEAD" = "1" ] && ARGS+=(--edl-head)

echo "[$(date)] eval ${INIT}/${STAGE}/${OBJ}  job=$SLURM_JOB_ID"
srun python -m vlfz.eval.run_eval "${ARGS[@]}"
