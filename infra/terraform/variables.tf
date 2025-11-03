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

variable "line_channel_access_token_param_name" {
  description = "SSM parameter name for LINE channel access token"
  type        = string
  default     = "/prod/line/CAT"
}

variable "line_channel_secret_param_name" {
  description = "SSM parameter name for LINE channel secret"
  type        = string
  default     = "/prod/line/SECRET"
}

variable "ingest_auth_token_param_name" {
  description = "SSM parameter name for ingest auth token"
  type        = string
  default     = "/prod/ingest/TOKEN"
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

