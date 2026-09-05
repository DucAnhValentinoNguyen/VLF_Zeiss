#!/bin/bash
# One-time stage of the curated WebDataset tier S3 -> LRZ. Run on a LOGIN NODE.
#
#   source lrz/job_env.sh          # exports AWS_PROFILE=gastronet-reader, GASTRONET_ROOT
#   bash pipeline/stage/stage_in.sh                 # sync everything
#   bash pipeline/stage/stage_in.sh shard_list.txt  # sync only the listed shards
#
# Idempotent (aws s3 sync skips unchanged); safe to re-run after an interruption.
set -euo pipefail
BUCKET="${GASTRONET_BUCKET:?set GASTRONET_BUCKET (terraform output bucket)}"
DEST="${GASTRONET_ROOT:?source lrz/job_env.sh first}/webdataset"
LIST="${1:-}"
mkdir -p "$DEST"

echo "[stage] s3://$BUCKET/webdataset/  ->  $DEST"
if [ -n "$LIST" ] && [ -f "$LIST" ]; then
  N=$(wc -l < "$LIST")
  echo "[stage] $N shards from $LIST"
  # sync excludes everything, re-includes each listed shard + its .json sidecar
  INCL=()
  while read -r s; do [ -n "$s" ] && INCL+=(--include "$s" --include "${s%.tar}.json"); done < "$LIST"
  aws s3 sync "s3://$BUCKET/webdataset/" "$DEST" --exclude "*" "${INCL[@]}" --no-progress
else
  aws s3 sync "s3://$BUCKET/webdataset/" "$DEST" --no-progress
fi

echo "[stage] verifying sizes against S3..."
python - "$BUCKET" "$DEST" <<'PY'
import os, sys, boto3
bucket, dest = sys.argv[1], sys.argv[2]
s3 = boto3.client("s3")
want = {}
p = s3.get_paginator("list_objects_v2")
for page in p.paginate(Bucket=bucket, Prefix="webdataset/"):
    for o in page.get("Contents", []):
        if o["Key"].endswith(".tar"):
            want[os.path.basename(o["Key"])] = o["Size"]
bad = 0
have = {f for f in os.listdir(dest) if f.endswith(".tar")}
for f, sz in want.items():
    if f not in have:
        continue
    got = os.path.getsize(os.path.join(dest, f))
    if got != sz:
        print(f"  SIZE MISMATCH {f}: local {got} != s3 {sz}"); bad += 1
n_local = len(have)
print(f"[stage] {n_local} local .tar shards; {bad} mismatch(es)")
sys.exit(1 if bad else 0)
PY

du -sh "$DEST"
echo "[stage] done. Point SSL at it:  CORPUS=gastronet GASTRONET_SOURCE=local_webdataset"
