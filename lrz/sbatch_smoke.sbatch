#!/bin/bash
#SBATCH --job-name=vlfz_smoke
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=00:40:00
#SBATCH --output=lrz/logs/%x_%j.out
#SBATCH --error=lrz/logs/%x_%j.err
#
# Full-codebase validation on a real GPU node: unit tests + LeJEPA/DINO SSL smoke
# (ViT-B, EMA teacher) + end-to-end HyperKvasir eval + zero-shot segmentation.
# No dataset downloads: SSL smoke uses the staged HKV-unlabeled pool.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source lrz/job_env.sh
wait_for_gpu || { echo "bad node"; exit 1; }

WORK="${SLURM_TMPDIR:-/tmp}/vlfz_smoke_$SLURM_JOB_ID"
mkdir -p "$WORK"
python -c "import torch; print('gpu:', torch.cuda.get_device_name(0), '| bf16:', torch.cuda.is_bf16_supported())"

echo "==== unit tests ===="
python -m pytest -q

echo "==== SSL smoke (LeJEPA + DINO, ViT-B) ===="
OUT_ROOT="$WORK/ssl" ALL=1 bash tests/smoke_ssl.sh

echo "==== eval smoke (HyperKvasir: pre + SSL + post + segmentation) ===="
SMK="$WORK/eval" bash tests/smoke_eval.sh

echo "==== smoke OK ===="
rm -rf "$WORK"
