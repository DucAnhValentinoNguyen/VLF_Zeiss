#!/bin/bash
# One-time: create the dedicated VLF_Zeiss venv with uv. Run on a login node.
#   bash lrz/setup_env.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh

echo "[setup] venv -> $VIRTUAL_ENV  (uv $(uv --version))"
uv venv --python 3.10 "$VIRTUAL_ENV"

# torch/torchvision from the CUDA 12.1 wheel index; everything else from PyPI.
uv pip install --python "$VIRTUAL_ENV/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1

uv pip install --python "$VIRTUAL_ENV/bin/python" -e ".[dev]"
# webdataset + boto3 (streaming/staging the S3 lake) and lightning (SSL training
# + eval feature extraction) are core deps, already in the line above.

# Lake-build tooling (pyarrow/matplotlib/prefect) — only needed to run the
# pipeline/{ingest,catalog,curate,dq}/ scripts from a login node; the ingest EC2
# ships its own via pipeline/requirements.txt.
uv pip install --python "$VIRTUAL_ENV/bin/python" -e ".[pipeline]"
# aws CLI on PATH (login node has none); rclone static binary for fast S3 sync.
uv pip install --python "$VIRTUAL_ENV/bin/python" awscli
mkdir -p "$HOME/bin"
if ! command -v rclone >/dev/null 2>&1; then
  ( cd /tmp && curl -fsSLO https://downloads.rclone.org/rclone-current-linux-amd64.zip \
    && unzip -oq rclone-current-linux-amd64.zip && cp rclone-*-linux-amd64/rclone "$HOME/bin/" \
    && chmod +x "$HOME/bin/rclone" && rm -rf rclone-*-linux-amd64* ) || echo "[setup] rclone fetch skipped"
fi

echo "[setup] sanity:"
"$VIRTUAL_ENV/bin/python" - <<'PY'
import torch, timm, open_clip, torchmetrics, sklearn, omegaconf
print("torch", torch.__version__, "cuda_build", torch.version.cuda)
print("timm", timm.__version__, "| open_clip", open_clip.__version__,
      "| torchmetrics", torchmetrics.__version__, "| sklearn", sklearn.__version__)
import vlfz; print("vlfz", vlfz.__version__, "importable")
PY
echo "[setup] done."
