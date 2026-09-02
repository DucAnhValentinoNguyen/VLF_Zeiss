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

echo "[setup] sanity:"
"$VIRTUAL_ENV/bin/python" - <<'PY'
import torch, timm, open_clip, torchmetrics, sklearn, omegaconf
print("torch", torch.__version__, "cuda_build", torch.version.cuda)
print("timm", timm.__version__, "| open_clip", open_clip.__version__,
      "| torchmetrics", torchmetrics.__version__, "| sklearn", sklearn.__version__)
import vlfz; print("vlfz", vlfz.__version__, "importable")
PY
echo "[setup] done."
