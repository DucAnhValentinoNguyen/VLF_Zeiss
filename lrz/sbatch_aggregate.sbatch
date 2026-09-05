#!/bin/bash
#SBATCH --job-name=vlfz_agg
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=lrz/logs/%x_%j.out
#SBATCH --error=lrz/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source lrz/job_env.sh

# aggregate is pure pandas/omegaconf, no GPU. Invoke the venv python by full
# path -- `srun` does not reliably carry job_env.sh's PATH edit onto the step.
PY="$VIRTUAL_ENV/bin/python"
echo "[$(date)] leakage audit + aggregate  ($PY)"
"$PY" -m vlfz.report.leakage_audit || echo "leakage audit skipped"
"$PY" -m vlfz.report.aggregate
echo "[$(date)] -> $OUT_ROOT/results/report.md"
