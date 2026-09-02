#!/bin/bash
# Obtain GastroNet-5M (~4.8M unlabeled PNGs, ~0.5-1 TB). GATED.
# RUN ON A LOGIN NODE.
#
# GastroNet-5M is NOT on HuggingFace. Access is via the dataset portal:
#     https://cortex.thetavision.nl/dataset-provider/listing/1/
# Register, accept the data-access agreement, and obtain the download URLs or a
# credentialed client. Then drop the *.zip shards (<=10k PNGs each) into:
#     $GASTRONET_ROOT   (default: $MCMLSCRATCH/gastronet5m)
#
# This script only: (1) checks quota, (2) resumably fetches from a URL list if you
# provide one, (3) builds the manifest vlfz needs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh

DEST="${GASTRONET_ROOT:-$MCMLSCRATCH/gastronet5m}"
URL_LIST="${1:-lrz/gastronet_urls.txt}"     # one https URL per line (from the portal)
mkdir -p "$DEST"

echo "[gastronet] target: $DEST"
df -h "$MCMLSCRATCH" | tail -1

if [ -f "$URL_LIST" ]; then
  echo "[gastronet] resumable download from $URL_LIST"
  wget -c -i "$URL_LIST" -P "$DEST" --no-verbose
else
  echo "[gastronet] no URL list at $URL_LIST -- place the *.zip shards in $DEST manually,"
  echo "            then re-run this script to build the manifest."
fi

N=$(ls "$DEST"/*.zip 2>/dev/null | wc -l || true)
echo "[gastronet] $N zip shards present"
[ "$N" -gt 0 ] && python -m vlfz.data.gastronet --build-manifest --force
echo "next: STAGE=full sbatch lrz/sbatch_ssl_pretrain.sbatch  (or lrz/submit_ssl_matrix.sh)"
