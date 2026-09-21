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
# Measured 2026-09-21: GPU util bursty 0-95% (avg low), 19/80GB VRAM used --
# I/O-bound on the single-worker portal decode path, not GPU-bound. cpus-per-
# task raised 8->16 alongside this to actually use the extra decode workers.
export PORTAL_TRANSFORM_WORKERS=8
source lrz/job_env.sh
