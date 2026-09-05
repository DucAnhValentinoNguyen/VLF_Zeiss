# Ephemeral in-region worker: ingests portal zips -> raw/, builds the Parquet
# catalog, curates the WebDataset tier, writes the DQ report, then (if
# auto_run_ingest) terminates itself. Same-region S3 traffic is free, so putting
# the heavy decode/resize/pack step here costs ~cents and zero egress.
#
# No inbound rules — you reach it with SSM Session Manager:
#   aws ssm start-session --target <instance_id> --region eu-north-1

resource "aws_security_group" "ingest" {
  name        = "gastronet-ingest"
  description = "Egress-only for the GastroNet ingest box"
  vpc_id     = data.aws_vpc.default.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

locals {
  user_data = templatefile("${path.module}/../ingest/user_data.sh", {
    bucket           = aws_s3_bucket.lake.bucket
    region           = var.region
    auto_run         = var.auto_run_ingest ? "1" : "0"
    portal_secret    = var.portal_secret_arn
    src_key          = aws_s3_object.pipeline_src.key
  })
}

resource "aws_launch_template" "ingest" {
  name_prefix   = "gastronet-ingest-"
  image_id      = data.aws_ssm_parameter.al2023.value
  instance_type = var.ingest_instance_type
  key_name      = var.ingest_key_name != "" ? var.ingest_key_name : null
  user_data     = base64encode(local.user_data)

  # All networking on the NIC (not instance-level) so aws_instance can stay
  # networking-free -- mixing a launch-template NIC with instance-level
  # security groups / subnet_id is rejected by RunInstances.
  network_interfaces {
    associate_public_ip_address = true
    delete_on_termination       = true
    security_groups             = [aws_security_group.ingest.id]
    subnet_id                   = data.aws_subnets.default.ids[0]
  }

  iam_instance_profile { arn = aws_iam_instance_profile.ingest.arn }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = var.ingest_root_gb
      volume_type           = "gp3"
      throughput            = 250
      iops                  = 6000
      delete_on_termination = true
      encrypted             = true
    }
  }

  dynamic "instance_market_options" {
    for_each = var.ingest_use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags          = { Name = "gastronet-ingest" }
  }
}

resource "aws_instance" "ingest" {
  launch_template {
    id      = aws_launch_template.ingest.id
    version = "$Latest"
  }

  # The box is meant to be recreated, not updated in place.
  lifecycle { ignore_changes = [launch_template, ami] }

  tags = { Name = "gastronet-ingest" }
}
