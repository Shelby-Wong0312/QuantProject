variable "project_name" { type = string }
variable "deps_layer_zip" { type = string }

variable "webhook_role_arn" { type = string }
variable "ingest_role_arn"  { type = string }
variable "push_role_arn"    { type = string }

variable "subscribers_table" { type = string }
variable "events_table"      { type = string }
variable "linegroups_table"  { type = string }
variable "systemstate_table" { type = string }

variable "line_access_token_param" { type = string }
variable "line_secret_param"       { type = string }
variable "ingest_token_param"      { type = string }

variable "trade_events_topic_arn" { type = string }

variable "tags" { type = map(string) default = {} }

data "archive_file" "common_layer" {
  type        = "zip"
  source_dir  = "${path.module}/../../../../line-trade-bot/src/common_layer/python"
  output_path = "${path.module}/../../build/common_layer.zip"
}

resource "aws_lambda_layer_version" "common" {
  layer_name          = "${var.project_name}-common"
  filename            = data.archive_file.common_layer.output_path
  compatible_runtimes = ["python3.11"]
}

resource "aws_lambda_layer_version" "deps" {
  layer_name          = "${var.project_name}-deps"
  filename            = var.deps_layer_zip
  compatible_runtimes = ["python3.11"]
}

data "archive_file" "webhook_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../../../line-trade-bot/src/webhook"
  output_path = "${path.module}/../../build/webhook.zip"
}

data "archive_file" "ingest_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../../../line-trade-bot/src/ingest"
  output_path = "${path.module}/../../build/ingest.zip"
}

data "archive_file" "push_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../../../line-trade-bot/src/line_push"
  output_path = "${path.module}/../../build/line_push.zip"
}

resource "aws_lambda_function" "webhook" {
  function_name = "${var.project_name}-webhook"
  role          = var.webhook_role_arn
  handler       = "app.lambda_handler"
  runtime       = "python3.11"
  filename      = data.archive_file.webhook_zip.output_path
  architectures = ["x86_64"]
  memory_size   = 256
  timeout       = 15
  layers        = [aws_lambda_layer_version.common.arn, aws_lambda_layer_version.deps.arn]
  environment {
    variables = {
      SUBSCRIBERS_TABLE         = var.subscribers_table
      EVENTS_TABLE              = var.events_table
      GROUPS_TABLE              = var.linegroups_table
      SYSTEM_STATE_TABLE        = var.systemstate_table
      STATE_TABLE               = var.systemstate_table
      LINE_CHANNEL_ACCESS_TOKEN_PARAM = var.line_access_token_param
      LINE_CHANNEL_SECRET_PARAM       = var.line_secret_param
    }
  }
  tags = var.tags
}

resource "aws_lambda_function" "ingest" {
  function_name = "${var.project_name}-ingest"
  role          = var.ingest_role_arn
  handler       = "app.lambda_handler"
  runtime       = "python3.11"
  filename      = data.archive_file.ingest_zip.output_path
  architectures = ["x86_64"]
  memory_size   = 256
  timeout       = 15
  layers        = [aws_lambda_layer_version.common.arn, aws_lambda_layer_version.deps.arn]
  environment {
    variables = {
      SUBSCRIBERS_TABLE         = var.subscribers_table
      EVENTS_TABLE              = var.events_table
      LINE_GROUPS_TABLE         = var.linegroups_table
      GROUPS_TABLE              = var.linegroups_table
      GROUP_WHITELIST_TABLE     = var.linegroups_table
      SYSTEM_STATE_TABLE        = var.systemstate_table
      LINE_CHANNEL_ACCESS_TOKEN_PARAM = var.line_access_token_param
      INGEST_TOKEN_PARAM              = var.ingest_token_param
    }
  }
  tags = var.tags
}

resource "aws_lambda_function" "line_push" {
  function_name = "${var.project_name}-line-push"
  role          = var.push_role_arn
  handler       = "app.lambda_handler"
  runtime       = "python3.11"
  filename      = data.archive_file.push_zip.output_path
  architectures = ["x86_64"]
  memory_size   = 256
  timeout       = 15
  layers        = [aws_lambda_layer_version.common.arn, aws_lambda_layer_version.deps.arn]
  environment {
    variables = {
      SUBSCRIBERS_TABLE         = var.subscribers_table
      EVENTS_TABLE              = var.events_table
      GROUPS_TABLE              = var.linegroups_table
      LINE_GROUPS_TABLE         = var.linegroups_table
      GROUP_WHITELIST_TABLE     = var.linegroups_table
      SYSTEM_STATE_TABLE        = var.systemstate_table
      LINE_CHANNEL_ACCESS_TOKEN_PARAM = var.line_access_token_param
    }
  }
  tags = var.tags
}

resource "aws_sns_topic_subscription" "line_push" {
  topic_arn = var.trade_events_topic_arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.line_push.arn
}

resource "aws_lambda_permission" "line_push_sns" {
  statement_id  = "AllowExecutionFromSNS"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.line_push.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = var.trade_events_topic_arn
}

output "webhook_function_arn"  { value = aws_lambda_function.webhook.arn }
output "webhook_function_name" { value = aws_lambda_function.webhook.function_name }
output "ingest_function_arn"   { value = aws_lambda_function.ingest.arn }
output "ingest_function_name"  { value = aws_lambda_function.ingest.function_name }

