data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  bucket = var.bucket_name != "" ? var.bucket_name : "gastronet5m-lake-${data.aws_caller_identity.current.account_id}-euno1"
}

# Default VPC + its subnets — the ingest box needs public egress to reach the
# portal and (free) S3. No inbound is opened.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Amazon Linux 2023, arch-matched to the instance family.
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}
