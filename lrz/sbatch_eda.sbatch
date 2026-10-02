#!/bin/bash
#SBATCH --job-name=vlfz_eda
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=lrz/logs/%x_%j.out
#SBATCH --error=lrz/logs/%x_%j.err
#
# Dataset EDA: exact HyperKvasir statistics, then a measured random sample of GastroNet-5M archives.
#   sbatch lrz/sbatch_eda.sbatch                 (N archives: EDA_N=24, workers: EDA_WORKERS=6)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source lrz/fullstream_env.sh
source .venv/bin/activate
export PORTAL_JSON="${PORTAL_JSON:-$HOME/.gastronet_portal/portal.json}"
echo "[$(date)] HyperKvasir EDA"
python scripts/eda_hyperkvasir.py
echo "[$(date)] GastroNet sample EDA"
python scripts/eda_gastronet_sample.py --n "${EDA_N:-24}" --workers "${EDA_WORKERS:-8}" --seed 0
echo "[$(date)] done"
