#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ "${CONFIRM:-0}" != "1" ]; then
  echo "Refusing to destroy compute without CONFIRM=1. This targets only aws_instance.ingest and aws_launch_template.ingest."
  exit 2
fi
python pipeline/flow.py down
