# VLF_Zeiss — GastroNet-5M data-lake infrastructure (AWS, eu-north-1).
#
# One `terraform apply` stands up: the lake S3 bucket (versioned, encrypted,
# private, lifecycle'd), an Athena workgroup + Glue tables over the Parquet
# catalog, an ephemeral in-region EC2 that ingests the portal zips + builds the
# curated WebDataset tier + writes a data-quality report and then self-
# terminates, a read-only managed policy for the LRZ cluster, and a cost budget
# alarm.
#
# Run from a laptop / admin box with AWS admin credentials — NOT from LRZ.
#   cd pipeline/infra
#   cp terraform.tfvars.example terraform.tfvars   # then edit
#   terraform init && terraform plan -out tf.plan
#   terraform apply tf.plan
#
# State is local by default (./terraform.tfstate, git-ignored). To use an S3
# backend instead, create a bucket yourself and uncomment the block below.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws     = { source = "hashicorp/aws", version = "~> 5.60" }
    archive = { source = "hashicorp/archive", version = "~> 2.4" }
  }

  # backend "s3" {
  #   bucket = "my-tfstate-bucket"
  #   key    = "vlf-zeiss/gastronet-lake.tfstate"
  #   region = "eu-north-1"
  # }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project   = "VLF_Zeiss"
      Component = "gastronet5m-data-lake"
      ManagedBy = "terraform"
    }
  }
}
