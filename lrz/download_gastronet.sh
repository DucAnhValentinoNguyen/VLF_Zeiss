#!/bin/bash
# Obtain GastroNet-5M for SSL pretraining.
#
# The dataset (~4.82M images, ~1 TB) does NOT fit LRZ. It lives in an AWS S3
# data lake; only a hard-curated 224px WebDataset tier (~40-60 GB) is staged here.
# See pipeline/README.md for the full flow. This script is just the last hop:
#
#   source lrz/job_env.sh                       # AWS_PROFILE + GASTRONET_BUCKET
#   bash lrz/download_gastronet.sh              # = pipeline/stage/stage_in.sh
#
# Legacy path (raw portal zips straight into $GASTRONET_ROOT, source=local_zip):
#   echo <url> per line > lrz/gastronet_urls.txt ; bash lrz/download_gastronet.sh --zips
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh

if [ "${1:-}" = "--zips" ]; then
  DEST="${GASTRONET_ROOT:?}"; URL_LIST="lrz/gastronet_urls.txt"; mkdir -p "$DEST"
  [ -f "$URL_LIST" ] || { echo "put shard URLs in $URL_LIST first"; exit 1; }
  wget -c -i "$URL_LIST" -P "$DEST" --no-verbose
  [ "$(ls "$DEST"/*.zip 2>/dev/null | wc -l)" -gt 0 ] && \
    GASTRONET_SOURCE=local_zip python -m vlfz.data.gastronet --build-manifest --force
  exit 0
fi

: "${GASTRONET_BUCKET:?set GASTRONET_BUCKET (terraform output bucket) or write ~/.gastronet_bucket}"
exec bash pipeline/stage/stage_in.sh "${1:-}"
