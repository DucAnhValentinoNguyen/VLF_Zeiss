output "bucket" {
  value       = aws_s3_bucket.lake.bucket
  description = "The lake bucket name."
}

output "bucket_arn" {
  value = aws_s3_bucket.lake.arn
}

output "gastronet_reader_policy_arn" {
  value       = aws_iam_policy.reader.arn
  description = "Attach to the LRZ IAM user: aws iam attach-user-policy --user-name read-only-agent --policy-arn <this>"
}

output "ingest_instance_id" {
  value       = aws_instance.ingest.id
  description = "SSM into it: aws ssm start-session --target <this> --region eu-north-1"
}

output "ingest_public_ip" {
  value = aws_instance.ingest.public_ip
}

output "athena_workgroup" {
  value = aws_athena_workgroup.gastronet.name
}

output "glue_database" {
  value = aws_glue_catalog_database.gastronet.name
}

output "stage_in_cmd" {
  value       = "AWS_PROFILE=gastronet-reader aws s3 sync s3://${aws_s3_bucket.lake.bucket}/webdataset/ $GASTRONET_ROOT/webdataset/"
  description = "Run on the LRZ login node once curation is done."
}
