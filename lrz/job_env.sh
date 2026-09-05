#!/bin/bash
# Common runtime environment for VLF_Zeiss jobs. Independent of any sibling repo.
# Source from every sbatch job AND from the submit_*.sh drivers:  source lrz/job_env.sh
# Only EXPORTS variables + resolves the HF token (no GPU calls) -> safe on a login node.

# --- repo root (this file lives in $VLF_ROOT/lrz/) ---
export VLF_ROOT="${VLF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- scratch storage (home is quota-limited; keep caches + outputs on scratch) ---
export MCMLSCRATCH="${MCMLSCRATCH:-/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra82sat2}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$MCMLSCRATCH/uv_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MCMLSCRATCH/.cache}"
export HF_HOME="${HF_HOME:-$MCMLSCRATCH/hf_cache}"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" 2>/dev/null || true

# --- dedicated venv for VLF_Zeiss (created by lrz/setup_env.sh) ---
export VIRTUAL_ENV="${VIRTUAL_ENV:-$VLF_ROOT/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# --- data: HyperKvasir is staged read-only on scratch; outputs go on the home
#     quota because the DSS project scratch container is full. ---
export DATA_ROOT="${DATA_ROOT:-$MCMLSCRATCH/zeiss_data}"
export HKV_ROOT="${HKV_ROOT:-$DATA_ROOT}"
export OUT_ROOT="${OUT_ROOT:-$HOME/vlf_zeiss_runs}"
export GASTRONET_ROOT="${GASTRONET_ROOT:-$HOME/vlf_zeiss_data/gastronet5m}"
export REALCOLON_ROOT="${REALCOLON_ROOT:-$HOME/vlf_zeiss_data/real_colon}"
mkdir -p "$OUT_ROOT" "$GASTRONET_ROOT" 2>/dev/null || true

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# import vlfz without installing (editable install also works; this is the fallback)
export PYTHONPATH="$VLF_ROOT:${PYTHONPATH:-}"

# --- AWS (read-only) for staging the curated GastroNet-5M tier from S3 ---
# Creds live in ~/.aws/credentials [gastronet-reader] (chmod 600, never committed).
# GASTRONET_BUCKET = `terraform output bucket`; drop it in ~/.gastronet_bucket.
export AWS_PROFILE="${AWS_PROFILE:-gastronet-reader}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-eu-north-1}"
[ -z "${GASTRONET_BUCKET:-}" ] && [ -f "$HOME/.gastronet_bucket" ] && \
  export GASTRONET_BUCKET="$(cat "$HOME/.gastronet_bucket")"
export GASTRONET_SOURCE="${GASTRONET_SOURCE:-local_webdataset}"
export PATH="$HOME/bin:$PATH"

# Robust GPU readiness check: torch.cuda.is_available() can transiently return False
# right after allocation on a busy shared node (NVML init race). Usage: wait_for_gpu || exit 1
wait_for_gpu() {
  for i in 1 2 3 4 5 6; do
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && return 0
    echo "  GPU/NVML not ready (attempt $i/6) -- retrying in 15s..."; sleep 15
  done
  echo "FATAL: no GPU after retries (resubmit -- likely a bad node)."; return 1
}

# --- HF token (needed for gated timm/open_clip SigLIP-2 tags on some mirrors) ---
if [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN="$(env -u HF_HOME -u HUGGING_FACE_HUB_TOKEN python -c \
    'from huggingface_hub import get_token; print(get_token() or "")' 2>/dev/null || true)"
  if [ -z "$HF_TOKEN" ]; then
    HF_TOKEN="$(cat "$HOME/.hf_token" 2>/dev/null || cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
  fi
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  printf '%s' "$HF_TOKEN" > "$HF_HOME/token" 2>/dev/null || true
fi
