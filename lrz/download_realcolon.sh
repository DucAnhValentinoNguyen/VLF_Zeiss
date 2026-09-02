#!/bin/bash
# Download the REAL-Colon dataset (~1 TB, CC BY 4.0) from Figshare.
# RUN ON A LOGIN NODE (no GPU needed). Takes many hours / days.
#
#   bash lrz/download_realcolon.sh
#
# Result layout under $REALCOLON_ROOT :
#   video_info.csv  lesion_info.csv
#   {SSS}-{VVV}_frames/*.jpg   {SSS}-{VVV}_annotations/*.xml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh

DEST="${REALCOLON_ROOT:-$MCMLSCRATCH/real_colon}"
HELPER="$MCMLSCRATCH/real-colon-dataset"
mkdir -p "$DEST"

echo "[realcolon] free space on target FS:"
df -h "$MCMLSCRATCH" | tail -1
echo "[realcolon] destination: $DEST  (need ~1 TB)"
read -r -p "continue? [y/N] " ok; [ "$ok" = "y" ] || exit 1

if [ ! -d "$HELPER/.git" ]; then
  git clone https://github.com/cosmoimd/real-colon-dataset "$HELPER"
fi
cd "$HELPER"

# The helper's figshare_dataset.py downloads every article part into ./dataset by
# default; point it at $DEST. Flags vary by helper version -- check --help.
python figshare_dataset.py --help || true
python figshare_dataset.py --output_folder "$DEST" || \
  python figshare_dataset.py "$DEST"

echo "[realcolon] done. sanity:"
ls "$DEST" | head
python - <<PY
import os
r = os.environ["REALCOLON_ROOT"] if "REALCOLON_ROOT" in os.environ else "$DEST"
vids = sorted(d[:-7] for d in os.listdir(r) if d.endswith("_frames"))
print("videos:", len(vids), vids[:3])
PY
echo "next: python -m vlfz.data.realcolon --inspect ; python -m vlfz.data.realcolon --make-splits"
