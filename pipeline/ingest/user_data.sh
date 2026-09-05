#!/bin/bash
# cloud-init for the ephemeral gastronet-ingest box. Terraform's templatefile()
# renders the five BUCKET/REGION/AUTO_RUN/PORTAL_SECRET_ARN/SRC_KEY values
# below; everything else lives in run_ingest.sh inside the shipped zip (no
# shell escaping headaches here).
set -eux
exec > >(tee -a /var/log/gastronet-ingest.log) 2>&1

BUCKET="${bucket}"
REGION="${region}"
AUTO_RUN="${auto_run}"
PORTAL_SECRET_ARN="${portal_secret}"
SRC_KEY="${src_key}"
export BUCKET REGION PORTAL_SECRET_ARN

dnf -y install python3.11 python3.11-pip unzip tar gzip awscli-2 >/dev/null 2>&1 || dnf -y install python3 python3-pip unzip tar gzip
python3.11 -m pip install --quiet --upgrade pip || true

mkdir -p /opt/gastronet && cd /opt/gastronet
aws s3 cp "s3://$BUCKET/$SRC_KEY" ./pipeline_src.zip --region "$REGION"
unzip -oq pipeline_src.zip -d src
python3.11 -m pip install --quiet -r src/pipeline/requirements.txt

cat > /opt/gastronet/env.sh <<EOF
export BUCKET="$BUCKET"
export REGION="$REGION"
export PORTAL_SECRET_ARN="$PORTAL_SECRET_ARN"
export PIPELINE_SRC=/opt/gastronet/src/pipeline
export PYTHONPATH=/opt/gastronet/src
EOF

if [ "$AUTO_RUN" = "1" ]; then
  bash /opt/gastronet/src/pipeline/ingest/run_ingest.sh --all --terminate
else
  echo "gastronet-ingest ready. SSM in and run:"
  echo "  source /opt/gastronet/env.sh && bash \$PIPELINE_SRC/ingest/run_ingest.sh --help"
fi
