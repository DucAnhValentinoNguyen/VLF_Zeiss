#!/bin/bash
#SBATCH --job-name=vlfz_smoke
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=00:40:00
#SBATCH --output=lrz/logs/%x_%j.out
#SBATCH --error=lrz/logs/%x_%j.err
#
# Full-codebase validation on a real GPU node: unit tests + LeJEPA/DINO SSL smoke
# (ViT-B, EMA teacher) + end-to-end eval on a synthetic REAL-Colon tree.
# No dataset downloads required (SSL smoke uses the staged HKV-unlabeled pool).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source lrz/job_env.sh
wait_for_gpu || { echo "bad node"; exit 1; }

python -c "import torch; print('gpu:', torch.cuda.get_device_name(0), '| bf16:', torch.cuda.is_bf16_supported())"

echo "==== unit tests ===="
python -m pytest -q

echo "==== SSL smoke (LeJEPA + DINO, ViT-B) ===="
SMK_SSL="$OUT_ROOT/_smoke/ssl"; rm -rf "$SMK_SSL"
OUT_ROOT="$SMK_SSL" ALL=1 bash tests/smoke_ssl.sh

echo "==== eval smoke (synthetic REAL-Colon, full pre/post/EDL) ===="
SMK_EVAL="$OUT_ROOT/_smoke/eval"; rm -rf "$SMK_EVAL"
SMK="$SMK_EVAL" bash tests/smoke_eval.sh

echo "==== smoke OK ===="
