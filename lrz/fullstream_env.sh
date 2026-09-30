#!/bin/bash
# Separate real home directory: never follow the historical runs symlink.
export OUT_ROOT="$HOME/vlf_zeiss_fullstream"
export STREAM_OBJECTIVES="${STREAM_OBJECTIVES:-dino}"
# Avoid W&B stale-run collisions while remaining stable across resubmissions.
export WANDB_RUN_NAMESPACE="${WANDB_RUN_NAMESPACE:-fullstream_optimized_v2}"
if [ "$STREAM_OBJECTIVES" = lejepa ]; then
  export PORTAL_CACHE="$OUT_ROOT/zip_cache_lejepa"
else
  export PORTAL_CACHE="$OUT_ROOT/zip_cache"
fi
export HF_HOME="$OUT_ROOT/model_cache"
export XDG_CACHE_HOME="$OUT_ROOT/cache_runtime"
export TORCH_HOME="$OUT_ROOT/model_cache/torch"
export UV_CACHE_DIR="$OUT_ROOT/uv_cache"
export WANDB_DIR="$OUT_ROOT/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export GASTRONET_ROOT="$OUT_ROOT/unused_local_corpus"
export GASTRONET_SOURCE=portal_zip
export INIT=imagenet CORPUS=gastronet STAGE=full SSL_EPOCHS=3 BATCH_SIZE=128
export BASE_LR=1e-4 MIN_LR=1e-6 LLRD=0.75 GRAD_CKPT=false
# Measured 2026-09-22, on cached archives + live portal range requests:
#   portal download   ~100 MiB/s on ONE connection (150 MiB/s at 4) -- NOT the
#                     bottleneck; production was only pulling ~20 MB/s.
#   decode            18.2 ms/image
#   +augmentation     47.2 ms/image lejepa, 85.4 ms/image dino  <-- the cost
# So throughput is CPU augmentation-bound and scales with worker count. At the
# old 4 workers lejepa managed 0.39 it/s (~60% of the 0.66 theoretical, the
# rest being IPC/batching/GPU). 12 workers fits the 16 allocated CPUs, leaving
# 4 for the main process, the download thread and GPU feed, and keeps archive
# processing (~60s) comfortably slower than the ~38s prefetch download so the
# single prefetch thread stays ahead.
export PORTAL_TRANSFORM_WORKERS=12
source lrz/job_env.sh
