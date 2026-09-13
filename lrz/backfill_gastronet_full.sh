#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh
PORTAL_JSON="${PORTAL_JSON:-$HOME/.gastronet_portal/portal.json}"
OUT_DIR="$GASTRONET_ROOT/webdataset"
[ -s "$PORTAL_JSON" ] || { echo "missing portal spec: $PORTAL_JSON" >&2; exit 2; }
mkdir -p "$OUT_DIR"
done_n=$(wc -l < "$OUT_DIR/_ingested.txt" 2>/dev/null || true)
echo "portal backfill: ${done_n:-0}/506 ingested; resuming in $OUT_DIR"
"$VIRTUAL_ENV/bin/python" -m pipeline.ingest.portal_to_wds \
  --portal-json "$PORTAL_JSON" --out "$OUT_DIR" --scratch "$MCMLSCRATCH/gnzips" \
  --limit 0 --jobs 8
