#!/bin/bash
# Drive the ingest box. Runs on the EC2 instance (Amazon Linux 2023), not LRZ.
#
#   source /opt/gastronet/env.sh
#   bash $PIPELINE_SRC/ingest/run_ingest.sh --ingest      # portal  -> s3://$BUCKET/raw/
#   bash $PIPELINE_SRC/ingest/run_ingest.sh --catalog     # raw/    -> catalog/manifest/
#   bash $PIPELINE_SRC/ingest/run_ingest.sh --curate      # raw/    -> webdataset/ + catalog/curated/
#   bash $PIPELINE_SRC/ingest/run_ingest.sh --dq          # catalog -> dq/data_report.md
#   bash $PIPELINE_SRC/ingest/run_ingest.sh --all --terminate
#
# Portal auth for --ingest (first match wins):
#   $PORTAL_SECRET_ARN   Secrets Manager JSON: {"cookie":..,"file_list":[..]} or {"url_list":[..]}
#   /opt/gastronet/portal.json   same JSON, dropped via `aws ssm` or scp
#   /opt/gastronet/urls.txt      plain https URL per line
set -euo pipefail
: "${BUCKET:?source env.sh first}" "${REGION:?}" "${PIPELINE_SRC:?}"
WORK=/mnt/work && mkdir -p "$WORK"
RAW="$WORK/raw" && mkdir -p "$RAW"
PY="python3.11"
DO_INGEST=0 DO_CATALOG=0 DO_CURATE=0 DO_DQ=0 TERMINATE=0

for a in "$@"; do case "$a" in
  --ingest) DO_INGEST=1;; --catalog) DO_CATALOG=1;; --curate) DO_CURATE=1;;
  --dq) DO_DQ=1;; --all) DO_INGEST=1; DO_CATALOG=1; DO_CURATE=1; DO_DQ=1;;
  --terminate) TERMINATE=1;;
  --help|-h) sed -n '2,20p' "$0"; exit 0;;
  *) echo "unknown arg $a"; exit 2;;
esac; done

if [ "$DO_INGEST" = 1 ]; then
  echo "== ingest: portal -> s3://$BUCKET/raw/ =="
  AUTH=()
  if [ -n "${PORTAL_SECRET_ARN:-}" ]; then
    aws secretsmanager get-secret-value --secret-id "$PORTAL_SECRET_ARN" \
      --region "$REGION" --query SecretString --output text > /opt/gastronet/portal.json
  fi
  [ -f /opt/gastronet/portal.json ] && AUTH+=(--portal-json /opt/gastronet/portal.json)
  [ -f /opt/gastronet/urls.txt ]    && AUTH+=(--url-list /opt/gastronet/urls.txt)
  $PY "$PIPELINE_SRC/ingest/portal_to_s3.py" \
      --bucket "$BUCKET" --raw-prefix raw/ --workdir "$RAW" "${AUTH[@]}"
fi

if [ "$DO_CATALOG" = 1 ]; then
  echo "== catalog: raw/ -> catalog/manifest/ =="
  $PY "$PIPELINE_SRC/catalog/build_manifest.py" \
      --bucket "$BUCKET" --raw-prefix raw/ --workdir "$WORK/cat" --upload
fi

if [ "$DO_CURATE" = 1 ]; then
  echo "== curate: raw/ -> webdataset/ + catalog/curated/ =="
  $PY "$PIPELINE_SRC/curate/curate_shard.py" \
      --bucket "$BUCKET" --raw-prefix raw/ --out-prefix webdataset/ \
      --curated-catalog catalog/curated/ --workdir "$WORK/cur" --jobs "$(nproc)" --upload
fi

if [ "$DO_DQ" = 1 ]; then
  echo "== dq: catalog -> dq/data_report.md =="
  $PY "$PIPELINE_SRC/dq/checks.py" --bucket "$BUCKET" --workdir "$WORK/dq" --upload
fi

echo "== done =="
if [ "$TERMINATE" = 1 ]; then
  ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id || true)
  echo "self-terminating $ID"
  aws ec2 terminate-instances --instance-ids "$ID" --region "$REGION" || shutdown -h now
fi
