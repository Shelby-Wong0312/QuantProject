terraform {
  required_version = ">= 1.4.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  name_prefix = var.project_name
  build_dir   = "${path.module}/build"
  # SSM parameter names
  ssm_access_token_name = var.line_channel_access_token_param_name
  ssm_secret_name       = var.line_channel_secret_param_name
  ssm_ingest_name       = var.ingest_auth_token_param_name
}

########################
# DynamoDB Tables
########################

resource "aws_dynamodb_table" "subscribers" {
  name         = "trade-events-subscribers"
  billing_mode = "PAY_PER_REQUEST"

  hash_key = "recipientId"
  attribute {
    name = "recipientId"
    type = "S"
  }
  sse_specification { enabled = true }
  tags = var.tags
}

resource "aws_dynamodb_table" "events" {
  name         = "TradeEvents"
  billing_mode = "PAY_PER_REQUEST"

  # Primary key: ts (ISO8601 string)
  hash_key = "ts"
  attribute { name = "ts" type = "S" }
  # GSI for recent N: partition by constant bucket=ALL, sort by ts desc
  attribute { name = "bucket" type = "S" }

  global_secondary_index {
    name               = "recent-index"
    hash_key           = "bucket"
    range_key          = "ts"
    projection_type    = "ALL"
    write_capacity     = null
    read_capacity      = null
  }

  sse_specification { enabled = true }
  tags = var.tags
}

resource "aws_dynamodb_table" "line_groups" {
  name         = "LineGroups"
  billing_mode = "PAY_PER_REQUEST"

  hash_key = "targetId"
  attribute { name = "targetId" type = "S" }
  sse_specification { enabled = true }
  tags = var.tags
}

resource "aws_dynamodb_table" "system_state" {
  name         = "SystemState"
  billing_mode = "PAY_PER_REQUEST"

  hash_key = "pk"
  attribute { name = "pk" type = "S" }
  sse_specification { enabled = true }
  tags = var.tags
}

########################
# SNS Topic
########################

resource "aws_sns_topic" "trade_events" {
  name = "trade-events"
  tags = var.tags
}

# Caller info for building ARNs
data "aws_caller_identity" "current" {}

locals {
  ssm_token_arn  = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${local.ssm_access_token_name}"
  ssm_secret_arn = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${local.ssm_secret_name}"
  ssm_ingest_arn = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${local.ssm_ingest_name}"
}

# Optional creation of SSM parameters (beware state stores values)
resource "aws_ssm_parameter" "line_channel_access_token" {
  count       = var.create_ssm_parameters ? 1 : 0
  name        = local.ssm_access_token_name
  description = "LINE channel access token"
  type        = "SecureString"
  value       = var.line_channel_access_token
  tags        = var.tags
}

resource "aws_ssm_parameter" "line_channel_secret" {
  count       = var.create_ssm_parameters ? 1 : 0
  name        = local.ssm_secret_name
  description = "LINE channel secret"
  type        = "SecureString"
  value       = var.line_channel_secret
  tags        = var.tags
}

resource "aws_ssm_parameter" "ingest_auth_token" {
  count       = var.create_ssm_parameters ? 1 : 0
  name        = local.ssm_ingest_name
  description = "Ingest auth token"
  type        = "SecureString"
  value       = var.ingest_auth_token
  tags        = var.tags
}

# Read LINE secrets from SSM for env injection
data "aws_ssm_parameter" "line_access_token" {
  name            = local.ssm_access_token_name
  with_decryption = true
}

data "aws_ssm_parameter" "line_secret" {
  name            = local.ssm_secret_name
  with_decryption = true
}

########################
# Lambda Layers
########################

data "archive_file" "common_layer" {
  type        = "zip"
  source_dir  = "${path.module}/../src/common_layer/python"
  output_path = "${local.build_dir}/common_layer.zip"
}

resource "aws_lambda_layer_version" "common" {
  layer_name          = "${local.name_prefix}-common"
  filename            = data.archive_file.common_layer.output_path
  compatible_runtimes = ["python3.11"]
  description         = "Common utilities"
}

resource "aws_lambda_layer_version" "deps" {
  layer_name          = "${local.name_prefix}-deps"
  filename            = var.deps_layer_zip
  compatible_runtimes = ["python3.11"]
  description         = "3rd-party deps (line-bot-sdk)"
}

########################
# Lambda packages
########################

data "archive_file" "webhook_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../src/webhook"
  output_path = "${local.build_dir}/webhook.zip"
}

data "archive_file" "ingest_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../src/ingest"
  output_path = "${local.build_dir}/ingest.zip"
}

data "archive_file" "line_push_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../src/line_push"
  output_path = "${local.build_dir}/line_push.zip"
}

########################
# IAM: role + log policy helper
########################

data "aws_iam_policy_document" "assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_cloudwatch_log_group" "webhook" {
  name              = "/aws/lambda/${local.name_prefix}-webhook"
  retention_in_days = 14
  tags              = var.tags
}

resource "aws_cloudwatch_log_group" "ingest" {
  name              = "/aws/lambda/${local.name_prefix}-ingest"
  retention_in_days = 14
  tags              = var.tags
}

resource "aws_cloudwatch_log_group" "line_push" {
  name              = "/aws/lambda/${local.name_prefix}-line-push"
  retention_in_days = 14
  tags              = var.tags
}

data "aws_iam_policy_document" "logs" {
  statement {
    actions = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = [
      aws_cloudwatch_log_group.webhook.arn,
      aws_cloudwatch_log_group.ingest.arn,
      aws_cloudwatch_log_group.line_push.arn,
      "${aws_cloudwatch_log_group.webhook.arn}:log-stream:*",
      "${aws_cloudwatch_log_group.ingest.arn}:log-stream:*",
      "${aws_cloudwatch_log_group.line_push.arn}:log-stream:*",
    ]
  }
}

resource "aws_iam_policy" "logs" {
  name   = "${local.name_prefix}-logs"
  policy = data.aws_iam_policy_document.logs.json
}

########################
# Lambda: webhook
########################

resource "aws_iam_role" "webhook" {
  name               = "${local.name_prefix}-webhook-role"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

data "aws_iam_policy_document" "webhook" {
  statement {
    actions = ["dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:GetItem"]
    resources = [aws_dynamodb_table.subscribers.arn]
  }
  statement {
    actions   = ["dynamodb:Query"]
    resources = ["${aws_dynamodb_table.events.arn}/index/recent-index"]
  }
  statement {
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem"]
    resources = [aws_dynamodb_table.events.arn]
  }
  statement {
    actions   = ["dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:GetItem"]
    resources = [aws_dynamodb_table.line_groups.arn]
  }
  statement {
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem"]
    resources = [aws_dynamodb_table.system_state.arn]
  }
  statement {
    actions   = ["ssm:GetParameter"]
    resources = [local.ssm_token_arn, local.ssm_secret_arn]
  }
}

resource "aws_iam_policy" "webhook" {
  name   = "${local.name_prefix}-webhook"
  policy = data.aws_iam_policy_document.webhook.json
}

resource "aws_iam_role_policy_attachment" "webhook_logs" {
  role       = aws_iam_role.webhook.name
  policy_arn = aws_iam_policy.logs.arn
}

resource "aws_iam_role_policy_attachment" "webhook_attach" {
  role       = aws_iam_role.webhook.name
  policy_arn = aws_iam_policy.webhook.arn
}

resource "aws_lambda_function" "webhook" {
  function_name = "${local.name_prefix}-webhook"
  role          = aws_iam_role.webhook.arn
  handler       = "app.lambda_handler"
  runtime       = "python3.11"
  filename      = data.archive_file.webhook_zip.output_path
  architectures = ["x86_64"]
  memory_size   = 256
  timeout       = 15
  layers        = [aws_lambda_layer_version.common.arn, aws_lambda_layer_version.deps.arn]
  environment {
    variables = {
      SUBSCRIBERS_TABLE        = aws_dynamodb_table.subscribers.name
      EVENTS_TABLE             = aws_dynamodb_table.events.name
      GROUPS_TABLE             = aws_dynamodb_table.line_groups.name
      SYSTEM_STATE_TABLE       = aws_dynamodb_table.system_state.name
      STATE_TABLE              = aws_dynamodb_table.system_state.name
      LINE_CHANNEL_ACCESS_TOKEN_PARAM = local.ssm_access_token_name
      LINE_CHANNEL_SECRET_PARAM       = local.ssm_secret_name
    }
  }
  tags = var.tags
}

########################
# Lambda: ingest (HTTP)
########################

resource "aws_iam_role" "ingest" {
  name               = "${local.name_prefix}-ingest-role"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

data "aws_iam_policy_document" "ingest" {
  statement {
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.events.arn]
  }
  statement {
    actions   = ["dynamodb:Scan", "dynamodb:GetItem"]
    resources = [aws_dynamodb_table.subscribers.arn]
  }
  statement {
    actions   = ["dynamodb:GetItem"]
    resources = [aws_dynamodb_table.line_groups.arn]
  }
  statement {
    actions   = ["dynamodb:PutItem", "dynamodb:GetItem"]
    resources = [aws_dynamodb_table.system_state.arn]
  }
  statement {
    actions   = ["ssm:GetParameter"]
    resources = [local.ssm_token_arn, local.ssm_ingest_arn]
  }
}

resource "aws_iam_policy" "ingest" {
  name   = "${local.name_prefix}-ingest"
  policy = data.aws_iam_policy_document.ingest.json
}

resource "aws_iam_role_policy_attachment" "ingest_logs" {
  role       = aws_iam_role.ingest.name
  policy_arn = aws_iam_policy.logs.arn
}

resource "aws_iam_role_policy_attachment" "ingest_attach" {
  role       = aws_iam_role.ingest.name
  policy_arn = aws_iam_policy.ingest.arn
}

resource "aws_lambda_function" "ingest" {
  function_name = "${local.name_prefix}-ingest"
  role          = aws_iam_role.ingest.arn
  handler       = "app.lambda_handler"
  runtime       = "python3.11"
  filename      = data.archive_file.ingest_zip.output_path
  architectures = ["x86_64"]
  memory_size   = 256
  timeout       = 15
  layers        = [aws_lambda_layer_version.common.arn, aws_lambda_layer_version.deps.arn]
  environment {
    variables = {
      SUBSCRIBERS_TABLE         = aws_dynamodb_table.subscribers.name
      EVENTS_TABLE              = aws_dynamodb_table.events.name
      LINE_CHANNEL_ACCESS_TOKEN_PARAM = local.ssm_access_token_name
      INGEST_TOKEN_PARAM              = local.ssm_ingest_name
      LINE_GROUPS_TABLE         = aws_dynamodb_table.line_groups.name
      GROUP_WHITELIST_TABLE     = aws_dynamodb_table.line_groups.name
      SYSTEM_STATE_TABLE        = aws_dynamodb_table.system_state.name
    }
  }
  tags = var.tags
}

########################
# Lambda: line-push (SNS)
########################

resource "aws_iam_role" "line_push" {
  name               = "${local.name_prefix}-line-push-role"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

data "aws_iam_policy_document" "line_push" {
  statement {
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.events.arn]
  }
  statement {
    actions   = ["dynamodb:Scan", "dynamodb:GetItem"]
    resources = [aws_dynamodb_table.subscribers.arn]
  }
  statement {
    actions   = ["dynamodb:GetItem"]
    resources = [aws_dynamodb_table.line_groups.arn]
  }
  statement {
    actions   = ["dynamodb:PutItem", "dynamodb:GetItem"]
    resources = [aws_dynamodb_table.system_state.arn]
  }
  statement {
    actions   = ["ssm:GetParameter"]
    resources = [local.ssm_token_arn]
  }
}

resource "aws_iam_policy" "line_push" {
  name   = "${local.name_prefix}-line-push"
  policy = data.aws_iam_policy_document.line_push.json
}

resource "aws_iam_role_policy_attachment" "line_push_logs" {
  role       = aws_iam_role.line_push.name
  policy_arn = aws_iam_policy.logs.arn
}

resource "aws_iam_role_policy_attachment" "line_push_attach" {
  role       = aws_iam_role.line_push.name
  policy_arn = aws_iam_policy.line_push.arn
}

resource "aws_lambda_function" "line_push" {
  function_name = "${local.name_prefix}-line-push"
  role          = aws_iam_role.line_push.arn
  handler       = "app.lambda_handler"
  runtime       = "python3.11"
  filename      = data.archive_file.line_push_zip.output_path
  architectures = ["x86_64"]
  memory_size   = 256
  timeout       = 15
  layers        = [aws_lambda_layer_version.common.arn, aws_lambda_layer_version.deps.arn]
  environment {
    variables = {
      SUBSCRIBERS_TABLE         = aws_dynamodb_table.subscribers.name
      EVENTS_TABLE              = aws_dynamodb_table.events.name
      GROUPS_TABLE              = aws_dynamodb_table.line_groups.name
      LINE_CHANNEL_ACCESS_TOKEN_PARAM = local.ssm_access_token_name
      LINE_GROUPS_TABLE         = aws_dynamodb_table.line_groups.name
      GROUP_WHITELIST_TABLE     = aws_dynamodb_table.line_groups.name
      SYSTEM_STATE_TABLE        = aws_dynamodb_table.system_state.name
    }
  }
  tags = var.tags
}

resource "aws_sns_topic_subscription" "line_push" {
  topic_arn = aws_sns_topic.trade_events.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.line_push.arn
}

resource "aws_lambda_permission" "line_push_sns" {
  statement_id  = "AllowExecutionFromSNS"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.line_push.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.trade_events.arn
}

########################
# API Gateway HTTP API
########################

resource "aws_apigatewayv2_api" "http" {
  name          = "${local.name_prefix}-api"
  protocol_type = "HTTP"
  cors_configuration {
    allow_origins     = ["*"]
    allow_methods     = ["POST", "OPTIONS"]
    allow_headers     = ["Content-Type", "X-Line-Signature", "X-Auth-Token"]
    max_age           = 86400
    allow_credentials = false
  }
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_apigatewayv2_integration" "webhook" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.webhook.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_integration" "ingest" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.ingest.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "webhook" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "POST /line/webhook"
  target    = "integrations/${aws_apigatewayv2_integration.webhook.id}"
}

resource "aws_apigatewayv2_route" "events" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "POST /events"
  target    = "integrations/${aws_apigatewayv2_integration.ingest.id}"
}

resource "aws_lambda_permission" "apigw_webhook" {
  statement_id  = "AllowAPIGatewayInvokeWebhook"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.webhook.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*/line/webhook"
}

resource "aws_lambda_permission" "apigw_ingest" {
  statement_id  = "AllowAPIGatewayInvokeIngest"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingest.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*/events"
}
