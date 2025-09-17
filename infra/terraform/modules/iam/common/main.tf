variable "aws_region" { type = string }
variable "line_channel_access_token_param" { type = string }
variable "line_channel_secret_param" { type = string }
variable "ingest_auth_token_param" { type = string }

data "aws_caller_identity" "current" {}

locals {
  ssm_access_token_arn = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${var.line_channel_access_token_param}"
  ssm_secret_arn       = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${var.line_channel_secret_param}"
  ssm_ingest_arn       = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${var.ingest_auth_token_param}"
}

output "ssm_access_token_arn" { value = local.ssm_access_token_arn }
output "ssm_secret_arn"       { value = local.ssm_secret_arn }
output "ssm_ingest_arn"       { value = local.ssm_ingest_arn }

