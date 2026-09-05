# Curation container — also usable as an AWS Batch (Fargate) array job.
# BUILD CONTEXT MUST BE pipeline/ (so requirements.txt + siblings are visible):
#   docker build -f pipeline/curate/Dockerfile -t gastronet-curate pipeline
#   aws ecr get-login-password --region eu-north-1 | docker login --username AWS \
#     --password-stdin <acct>.dkr.ecr.eu-north-1.amazonaws.com
#   docker tag gastronet-curate <acct>.dkr.ecr.eu-north-1.amazonaws.com/gastronet-curate:latest
#   docker push  <acct>.dkr.ecr.eu-north-1.amazonaws.com/gastronet-curate:latest
#
# As a Batch array job, set --only-group "g$(printf '%03d' $AWS_BATCH_JOB_ARRAY_INDEX)".
FROM public.ecr.aws/docker/library/python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libjpeg62-turbo libwebp7 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/pipeline
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "/app/pipeline/curate/curate_shard.py"]
