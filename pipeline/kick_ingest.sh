#!/bin/bash
# Ship the pipeline code + portal session to the lake, then tell the ephemeral
# EC2 to run ingest -> catalog -> curate -> dq. Run from a shell that has ADMIN
# AWS creds exported (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
#
#   export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_DEFAULT_REGION=eu-north-1
#   bash pipeline/kick_ingest.sh [N_ZIPS]        # N_ZIPS default 60 (~250 GB subset); 0 = all 506
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUCKET=gastronet5m-lake-709569057971-euno1
REGION=eu-north-1
PORTAL_JSON="$HOME/.gastronet_portal/portal.json"
LIMIT="${1:-60}"

command -v aws >/dev/null || { echo "aws CLI not on PATH"; exit 1; }
aws sts get-caller-identity >/dev/null || { echo "no AWS creds -- export admin keys first"; exit 1; }
[ -f "$PORTAL_JSON" ] || { echo "missing $PORTAL_JSON"; exit 1; }

echo "== 1/3  re-ship pipeline code (terraform re-uploads bootstrap/pipeline_src.zip) =="
( cd pipeline/infra && terraform apply -auto-approve -target=aws_s3_object.pipeline_src )
INSTANCE=$(cd pipeline/infra && terraform output -raw ingest_instance_id)
echo "   ingest instance: $INSTANCE"

echo "== 2/3  upload portal session =="
aws s3 cp "$PORTAL_JSON" "s3://$BUCKET/bootstrap/portal.json"

echo "== 3/3  start ingest on $INSTANCE (limit=$LIMIT) =="
echo "   waiting for SSM to register the instance..."
for i in $(seq 1 40); do
  st=$(aws ssm describe-instance-information --region "$REGION" \
        --filters "Key=InstanceIds,Values=$INSTANCE" \
        --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [ "$st" = "Online" ] && { echo "   SSM online"; break; }
  sleep 15
done
read -r -d '' REMOTE <<EOF || true
set -eux
for i in \$(seq 1 40); do [ -f /opt/gastronet/env.sh ] && break; echo "waiting for cloud-init..."; sleep 15; done
cd /opt/gastronet
aws s3 cp s3://$BUCKET/bootstrap/pipeline_src.zip pipeline_src.zip
rm -rf src && unzip -oq pipeline_src.zip -d src
source /opt/gastronet/env.sh
export INGEST_LIMIT=$LIMIT
nohup bash \$PIPELINE_SRC/ingest/run_ingest.sh --ingest --catalog --curate --dq \
  > /var/log/gastronet-ingest-run.log 2>&1 &
echo "started pid \$!"
EOF

PARAMS=$(mktemp)
python3 -c 'import json,sys; json.dump({"commands":[sys.stdin.read()],"executionTimeout":["172800"]}, open(sys.argv[1],"w"))' "$PARAMS" <<<"$REMOTE"
CID=$(aws ssm send-command --region "$REGION" --instance-ids "$INSTANCE" \
  --document-name AWS-RunShellScript --parameters "file://$PARAMS" \
  --query 'Command.CommandId' --output text)
rm -f "$PARAMS"
echo "SSM command: $CID"
echo
echo "watch it:"
echo "  aws ssm send-command --region $REGION --instance-ids $INSTANCE \\"
echo "    --document-name AWS-RunShellScript \\"
echo "    --parameters 'commands=[\"tail -40 /var/log/gastronet-ingest-run.log\"]' \\"
echo "    --query Command.CommandId --output text"
echo "  # then: aws ssm get-command-invocation --region $REGION --command-id <id> --instance-id $INSTANCE --query StandardOutputContent --output text"
echo
echo "or just watch the bucket fill from LRZ (read-only key):"
echo "  AWS_PROFILE=gastronet-reader aws s3 ls s3://$BUCKET/raw/ | wc -l"
