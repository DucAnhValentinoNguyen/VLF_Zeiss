# ---------------------------------------------------------------- the lake
resource "aws_s3_bucket" "lake" {
  bucket = local.bucket
}

resource "aws_s3_bucket_versioning" "lake" {
  bucket = aws_s3_bucket.lake.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "lake" {
  bucket                  = aws_s3_bucket.lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id

  # Raw portal zips: pristine copy, rarely read after curation -> cold storage.
  rule {
    id     = "raw-to-cold"
    status = "Enabled"
    filter { prefix = "raw/" }
    transition {
      days          = var.raw_to_glacier_days
      storage_class = var.raw_storage_class
    }
    noncurrent_version_expiration { noncurrent_days = 30 }
  }

  # Never leave half-finished multipart uploads billing.
  rule {
    id     = "abort-mpu"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload { days_after_initiation = 7 }
  }

  # Athena scratch is disposable.
  rule {
    id     = "expire-athena-results"
    status = "Enabled"
    filter { prefix = "athena-results/" }
    expiration { days = 14 }
  }
}

# One place for every prefix the pipeline uses, so the layout is self-documenting.
resource "aws_s3_object" "prefixes" {
  for_each = toset([
    "raw/", "webdataset/", "catalog/manifest/", "catalog/curated/",
    "ingest_log/", "dq/", "bootstrap/", "athena-results/",
  ])
  bucket  = aws_s3_bucket.lake.id
  key     = each.value
  content = ""
}

# ---- pipeline code shipped to the ingest box -----------------------------
data "archive_file" "pipeline_src" {
  type        = "zip"
  source_dir  = "${path.module}/.."
  output_path = "${path.module}/.build/pipeline_src.zip"
  excludes    = ["infra/.terraform", "infra/.build", "infra/tf.plan", "infra/terraform.tfstate", "infra/terraform.tfstate.backup"]
}

resource "aws_s3_object" "pipeline_src" {
  bucket = aws_s3_bucket.lake.id
  key    = "bootstrap/pipeline_src.zip"
  source = data.archive_file.pipeline_src.output_path
  etag   = data.archive_file.pipeline_src.output_md5
}
