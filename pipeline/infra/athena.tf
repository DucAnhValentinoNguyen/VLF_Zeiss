resource "aws_glue_catalog_database" "gastronet" {
  name = "gastronet"
}

resource "aws_athena_workgroup" "gastronet" {
  name          = "gastronet"
  force_destroy = true
  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = false
    result_configuration {
      output_location = "s3://${aws_s3_bucket.lake.bucket}/athena-results/"
      encryption_configuration { encryption_option = "SSE_S3" }
    }
  }
}

# Raw-corpus catalog: one row per image inside the portal zips.
resource "aws_glue_catalog_table" "manifest" {
  name          = "manifest"
  database_name = aws_glue_catalog_database.gastronet.name
  table_type    = "EXTERNAL_TABLE"
  parameters = {
    classification        = "parquet"
    "parquet.compression" = "SNAPPY"
    EXTERNAL              = "TRUE"
  }
  partition_keys {
    name = "shard_group"
    type = "string"
  }
  storage_descriptor {
    location      = "s3://${aws_s3_bucket.lake.bucket}/catalog/manifest/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    columns {
      name = "shard"
      type = "string"
    }
    columns {
      name = "member"
      type = "string"
    }
    columns {
      name = "bytes"
      type = "bigint"
    }
    columns {
      name = "sha256"
      type = "string"
    }
    columns {
      name = "ext"
      type = "string"
    }
    columns {
      name = "width"
      type = "int"
    }
    columns {
      name = "height"
      type = "int"
    }
    columns {
      name = "mode"
      type = "string"
    }
    columns {
      name = "phash"
      type = "string"
    }
    columns {
      name = "corrupt"
      type = "boolean"
    }
  }
}

# Curated-tier catalog: one row per image kept in the WebDataset shards.
resource "aws_glue_catalog_table" "curated" {
  name          = "curated"
  database_name = aws_glue_catalog_database.gastronet.name
  table_type    = "EXTERNAL_TABLE"
  parameters = {
    classification        = "parquet"
    "parquet.compression" = "SNAPPY"
    EXTERNAL              = "TRUE"
  }
  partition_keys {
    name = "shard_group"
    type = "string"
  }
  storage_descriptor {
    location      = "s3://${aws_s3_bucket.lake.bucket}/catalog/curated/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    columns {
      name = "wds_shard"
      type = "string"
    }
    columns {
      name = "key"
      type = "string"
    }
    columns {
      name = "src_shard"
      type = "string"
    }
    columns {
      name = "src_member"
      type = "string"
    }
    columns {
      name = "bytes"
      type = "bigint"
    }
    columns {
      name = "width"
      type = "int"
    }
    columns {
      name = "height"
      type = "int"
    }
    columns {
      name = "phash"
      type = "string"
    }
    columns {
      name = "dup_of"
      type = "string"
    }
  }
}
