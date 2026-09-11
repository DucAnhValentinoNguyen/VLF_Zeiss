# GastroNet-5M data pipeline

A reproducible path that gets **all ~4.82 M** unlabeled GastroNet-5M images from
the `cortex.thetavision.nl` portal into a form the four SSL pretraining jobs on
the LRZ H100 cluster can actually consume — without the ~1 TB ever needing to fit
LRZ's ~90 GB of writable quota.

```
 cortex.thetavision.nl            AWS  eu-north-1  (the data lake)                 LRZ cluster
 ────────────────────    ┌───────────────────────────────────────────────┐    ──────────────────
  506 gated .zip shards  │  s3://…-lake/                                  │
        (~1 TB)          │    raw/               506 zips, sha256'd       │
          │  ephemeral   │      │  build_manifest.py                      │
          │  EC2 (spot,  │      ▼                                         │
          └──rclone──────┼──► catalog/manifest/  Parquet, 1 row/image ───┼─► Athena  (count,
                         │      │  curate_shard.py   (resize 224 / JPEG   │    dedup %, dims…)
                         │      │   q87 / phash-dedup / pack)             │
                         │      ▼                                         │
                         │    webdataset/        ~40–60 GB, ~1 GB .tars ──┼─► stage_in.sh  (one
                         │    catalog/curated/   Parquet, 1 row/kept img  │    `aws s3 sync`,
                         │    dq/data_report.md                          │    ~$5 egress, once)
                         │                                               │        │
                         └───────────────────────────────────────────────┘        ▼
                                                                          $GASTRONET_ROOT/webdataset/
                                                                          → 4× SSL jobs read local disk,
                                                                            every epoch, all 4.82 M imgs
```

**Design rule:** the bucket is the durable lake; LRZ gets the curated tier **once**
and every training epoch reads local disk. No per-epoch S3 reads — that would be
hundreds of dollars of egress. A `s3_webdataset` streaming mode exists
(`vlfz/data/gastronet.py`) for when the curated tier outgrows local disk, with
the per-pass egress cost called out.

## Layout

| path | what |
|---|---|
| `infra/` | Terraform — the whole lake in one `apply`: S3 (versioned, SSE, lifecycle, private), Glue DB + Athena workgroup + external tables, ephemeral ingest/curation EC2 (spot, SSM-driven, self-terminating), a read-only IAM policy for LRZ, a cost-budget alarm |
| `ingest/` | `portal_to_s3.py` — idempotent, checksummed portal → `raw/`; `run_ingest.sh` / `user_data.sh` drive the EC2 box |
| `catalog/` | `build_manifest.py` — every zip's contents → partitioned Parquet catalog |
| `curate/` | `curate_shard.py` (+ `Dockerfile` for AWS Batch scale-out) — `raw/` → `webdataset/` + curated catalog |
| `stage/` | `select_subset.py` (optional narrowing) + `stage_in.sh` (S3 → LRZ, sha256-verified) |
| `dq/` | `checks.py` → `data_report.md` + histograms; a non-zero corrupt rate or wild count stops the line |
| `flow.py`, `Makefile` | orchestration — `terraform apply` → SSM step sequence → `status`; Prefect flow if installed, plain runner otherwise |

## Run it

Prereqs on your laptop: Terraform ≥ 1.5, AWS CLI with **admin** creds for the
target account (`aws sts get-caller-identity`), Docker only if you want the Batch
path.

```bash
cd pipeline/infra
cp terraform.tfvars.example terraform.tfvars      # set alert_email; pick storage class
terraform init && terraform plan -out tf.plan     # review — commit the plan as evidence
terraform apply tf.plan                            # ~2 min; prints bucket + instance id

# put portal auth on the box (one of):
#   aws ssm start-session --target <id> --region eu-north-1
#   # then: printf '%s' '<cookie>' ...  or drop /opt/gastronet/urls.txt / portal.json
# or store it as a Secrets Manager secret and set portal_secret_arn in tfvars.

python ../flow.py run --step ingest      # portal  -> raw/           (~1 TB, hours)
python ../flow.py run --step catalog     # raw/    -> catalog/manifest/
#   Athena:  MSCK REPAIR TABLE gastronet.manifest;
#            SELECT count(*), sum(bytes), count(*) FILTER (WHERE corrupt) FROM gastronet.manifest;
python ../flow.py run --step curate      # raw/    -> webdataset/ + catalog/curated/
#   Athena:  MSCK REPAIR TABLE gastronet.curated;
python ../flow.py run --step dq          # -> s3://…/dq/data_report.md   (inspect!)
python ../flow.py down                   # terminate the EC2 box (bucket + data stay)
```

Then, **on an LRZ login node**:

```bash
echo "<bucket-from-terraform-output>" > ~/.gastronet_bucket
AWS_PROFILE=gastronet-reader aws iam attach-user-policy \
  --user-name read-only-agent --policy-arn "<gastronet_reader_policy_arn output>"   # once, needs admin
source lrz/job_env.sh
bash pipeline/stage/stage_in.sh                 # s3 -> $GASTRONET_ROOT/webdataset/  (~50 GB, ~$5)
python -m vlfz.data.gastronet --source local_webdataset --smoke   # iterate a few batches

# SSL now has its corpus:
CORPUS=gastronet GASTRONET_SOURCE=local_webdataset bash lrz/submit_ssl_matrix.sh
```

## Cost (full corpus, `eu-north-1`)

| item | one-time | monthly |
|---|--:|--:|
| S3 ingress (portal → bucket) | $0 | — |
| ingest/curation EC2 (c7i.2xlarge spot, ~6 h) | ~$1 | — |
| `raw/` ~1 TB, first 30 d in Standard | ~$8 | — |
| `raw/` ~1 TB in Glacier IR after lifecycle | — | ~$4 |
| `webdataset/` ~50 GB Standard | — | ~$1.2 |
| Athena (Parquet, partitioned) | — | ~$0 |
| one-time curated stage-in egress to LRZ (~50 GB) | ~$5 | — |
| **total** | **~$25–35** | **~$5** |

Well inside the $100 AWS credit. `terraform destroy` (or `make destroy-all`)
returns to zero. AWS Budgets emails at 40 % and 100 % of the monthly cap.

## Adding additional datasets later

Point `ingest/portal_to_s3.py` at raw archives (`--url-list`), reuse
`build_manifest.py` / `curate_shard.py` unchanged (they are format-agnostic over
`raw/*.zip|*.tar`), and either stage a curated subset or switch that corpus to
`s3_webdataset`.
