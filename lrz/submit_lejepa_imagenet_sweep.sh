#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh
mkdir -p lrz/logs
for spec in \
  "s60_lr5e-5|5e-5|5e-7|0.75" \
  "s60_llrd0p65|1e-4|1e-6|0.65" \
  "s60_lr5e-5__llrd0p65|5e-5|5e-7|0.65"; do
  IFS='|' read -r tag lr min llrd <<< "$spec"
  echo "queue $tag: base_lr=$lr min_lr=$min llrd=$llrd"
  BASE_LR="$lr" MIN_LR="$min" LLRD="$llrd" RUN_TAG="$tag" OBJ=lejepa INIT=imagenet \
    CORPUS=gastronet STAGE=full SSL_EXTRA="--source local_webdataset" \
    sbatch --export=ALL lrz/sbatch_ssl_pretrain.sbatch
done
