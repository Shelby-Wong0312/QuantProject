variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "project_name" {
  description = "Project name prefix"
  type        = string
  default     = "line-trade-bot"
}

variable "line_channel_access_token" {
  description = "LINE Messaging API Channel Access Token (used only when create_ssm_parameters=true)"
  type        = string
  sensitive   = true
  default     = null
}

variable "line_channel_secret" {
  description = "LINE Messaging API Channel Secret (used only when create_ssm_parameters=true)"
  type        = string
  sensitive   = true
  default     = null
}

variable "ingest_auth_token" {
  description = "Shared secret for /events ingestion endpoint (used only when create_ssm_parameters=true)"
  type        = string
  sensitive   = true
  default     = null
}

variable "create_ssm_parameters" {
  description = "Create SSM parameters for secrets (will store values in Terraform state)"
  type        = bool
  default     = false
}

variable "ssm_prefix" {
  description = "SSM parameter prefix"
  type        = string
  default     = "/line-trade-bot"
}

variable "line_channel_access_token_param_name" {
  description = "SSM parameter name for LINE channel access token"
  type        = string
  default     = "/line-trade-bot/line_channel_access_token"
}

variable "line_channel_secret_param_name" {
  description = "SSM parameter name for LINE channel secret"
  type        = string
  default     = "/line-trade-bot/line_channel_secret"
}

variable "ingest_auth_token_param_name" {
  description = "SSM parameter name for ingest auth token"
  type        = string
  default     = "/line-trade-bot/ingest_auth_token"
}

variable "deps_layer_zip" {
  description = "Path to prebuilt dependencies layer zip (contains python/ with line-bot-sdk)"
  type        = string
  default     = "${path.module}/build/deps_layer.zip"
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}
