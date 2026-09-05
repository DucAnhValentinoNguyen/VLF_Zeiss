# ---------------------------------------------------------------- ingest role
# Attached to the ephemeral EC2. Read/write ONLY inside the lake bucket, plus
# SSM so you can drive the box without SSH, plus Secrets Manager read for the
# optional portal-auth secret.
data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ingest" {
  name               = "gastronet-ingest"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

data "aws_iam_policy_document" "ingest" {
  statement {
    sid       = "LakeBucketList"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.lake.arn]
  }
  statement {
    sid       = "LakeObjectsRW"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    resources = ["${aws_s3_bucket.lake.arn}/*"]
  }
  dynamic "statement" {
    for_each = var.portal_secret_arn != "" ? [1] : []
    content {
      sid       = "PortalSecret"
      actions   = ["secretsmanager:GetSecretValue"]
      resources = [var.portal_secret_arn]
    }
  }
}

resource "aws_iam_role_policy" "ingest" {
  name   = "gastronet-ingest-inline"
  role   = aws_iam_role.ingest.id
  policy = data.aws_iam_policy_document.ingest.json
}

resource "aws_iam_role_policy_attachment" "ingest_ssm" {
  role       = aws_iam_role.ingest.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ingest" {
  name = "gastronet-ingest"
  role = aws_iam_role.ingest.name
}

# ---------------------------------------------------------------- reader policy
# Read-only over the lake + Athena. ATTACH THIS to the pre-existing
# `read-only-agent` IAM user (whose access key already lives on LRZ):
#   aws iam attach-user-policy --user-name read-only-agent \
#     --policy-arn <gastronet_reader_policy_arn output>
data "aws_iam_policy_document" "reader" {
  statement {
    sid       = "LakeList"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.lake.arn]
  }
  statement {
    sid       = "LakeRead"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.lake.arn}/*"]
  }
  statement {
    sid = "AthenaRead"
    actions = [
      "athena:StartQueryExecution", "athena:GetQueryExecution",
      "athena:GetQueryResults", "athena:GetWorkGroup", "athena:StopQueryExecution",
      "glue:GetDatabase", "glue:GetDatabases", "glue:GetTable", "glue:GetTables",
      "glue:GetPartition", "glue:GetPartitions",
    ]
    resources = ["*"]
  }
  statement {
    sid       = "AthenaResultsRW"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.lake.arn}/athena-results/*"]
  }
}

resource "aws_iam_policy" "reader" {
  name        = "gastronet-lake-ro"
  description = "Read-only access to the GastroNet-5M lake + Athena, for LRZ."
  policy      = data.aws_iam_policy_document.reader.json
}

# Uncomment to let Terraform own the attachment to your existing user:
# resource "aws_iam_user_policy_attachment" "reader" {
#   user       = "read-only-agent"
#   policy_arn = aws_iam_policy.reader.arn
# }
