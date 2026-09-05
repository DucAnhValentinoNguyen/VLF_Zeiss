variable "region" {
  type    = string
  default = "eu-north-1"
}

variable "bucket_name" {
  type        = string
  default     = ""
  description = "Lake bucket name. Empty -> gastronet5m-lake-<account_id>-euno1."
}

variable "alert_email" {
  type        = string
  description = "Email for the cost-budget alarm (a confirmation mail is sent once)."
}

variable "monthly_budget_usd" {
  type    = number
  default = 60
}

variable "raw_to_glacier_days" {
  type        = number
  default     = 30
  description = "Age at which raw/ objects move to Glacier Instant Retrieval."
}

variable "raw_storage_class" {
  type        = string
  default     = "GLACIER_IR"
  description = "GLACIER_IR (~$4/TB/mo, ms retrieval) or DEEP_ARCHIVE (~$1/TB/mo, hours)."
}

# ---- ingest / curation EC2 -------------------------------------------------
variable "ingest_instance_type" {
  type    = string
  default = "c7i.2xlarge" # 8 vCPU / 16 GiB — decode+resize+pack the whole corpus in a few hours
}

variable "ingest_use_spot" {
  type    = bool
  default = true
}

variable "ingest_root_gb" {
  type        = number
  default     = 1200
  description = "Root EBS size — must hold the ~1 TB of raw zips while curating."
}

variable "ingest_key_name" {
  type        = string
  default     = ""
  description = "Optional EC2 key pair name for SSH debugging. Empty -> no SSH key (SSM only)."
}

variable "auto_run_ingest" {
  type        = bool
  default     = false
  description = "If true the instance runs the full pipeline on boot then terminates. If false it just boots and waits (you drive it over SSM)."
}

variable "portal_secret_arn" {
  type        = string
  default     = ""
  description = "Optional Secrets Manager ARN holding portal auth (cookie/token/url-list JSON). Empty -> supply auth over SSM at run time."
}
