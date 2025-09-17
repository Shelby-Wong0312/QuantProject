variable "project_name" { type = string }
variable "subscribers_table_arn" { type = string }
variable "events_table_arn" { type = string }
variable "linegroups_table_arn" { type = string }
variable "systemstate_table_arn" { type = string }
variable "line_access_token_param_arn" { type = string }
variable "line_secret_param_arn" { type = string }
variable "ingest_token_param_arn" { type = string }

data "aws_iam_policy_document" "assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals { type = "Service" identifiers = ["lambda.amazonaws.com"] }
  }
}

# CloudWatch Logs minimal write policy (attach per role via inline)
data "aws_iam_policy_document" "logs_write" {
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["*"]
  }
}

# webhook role (read/write three tables + read SSM params)
data "aws_iam_policy_document" "webhook_inline" {
  statement { actions = ["dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:GetItem"] resources = [var.subscribers_table_arn] }
  statement { actions = ["dynamodb:GetItem", "dynamodb:PutItem"] resources = [var.events_table_arn] }
  statement { actions = ["dynamodb:Query"] resources = ["${var.events_table_arn}/index/recent-index"] }
  statement { actions = ["dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:GetItem"] resources = [var.linegroups_table_arn] }
  statement { actions = ["dynamodb:GetItem", "dynamodb:PutItem"] resources = [var.systemstate_table_arn] }
  statement { actions = ["ssm:GetParameter"] resources = [var.line_access_token_param_arn, var.line_secret_param_arn] }
}

resource "aws_iam_role" "webhook" {
  name               = "${var.project_name}-webhook-role"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

resource "aws_iam_role_policy" "webhook_logs" { role = aws_iam_role.webhook.id policy = data.aws_iam_policy_document.logs_write.json }
resource "aws_iam_role_policy" "webhook_inline" { role = aws_iam_role.webhook.id policy = data.aws_iam_policy_document.webhook_inline.json }

# ingest role (write events, read subscribers/linegroups, read/write systemstate, read SSM for token)
data "aws_iam_policy_document" "ingest_inline" {
  statement { actions = ["dynamodb:PutItem"] resources = [var.events_table_arn] }
  statement { actions = ["dynamodb:Scan", "dynamodb:GetItem"] resources = [var.subscribers_table_arn] }
  statement { actions = ["dynamodb:GetItem"] resources = [var.linegroups_table_arn] }
  statement { actions = ["dynamodb:PutItem", "dynamodb:GetItem"] resources = [var.systemstate_table_arn] }
  statement { actions = ["ssm:GetParameter"] resources = [var.line_access_token_param_arn, var.ingest_token_param_arn] }
}

resource "aws_iam_role" "ingest" {
  name               = "${var.project_name}-ingest-role"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

resource "aws_iam_role_policy" "ingest_logs" { role = aws_iam_role.ingest.id policy = data.aws_iam_policy_document.logs_write.json }
resource "aws_iam_role_policy" "ingest_inline" { role = aws_iam_role.ingest.id policy = data.aws_iam_policy_document.ingest_inline.json }

# line-push role (sns-triggered; write events, read subscribers/linegroups, rw systemstate, read SSM for token)
data "aws_iam_policy_document" "push_inline" {
  statement { actions = ["dynamodb:PutItem"] resources = [var.events_table_arn] }
  statement { actions = ["dynamodb:Scan", "dynamodb:GetItem"] resources = [var.subscribers_table_arn] }
  statement { actions = ["dynamodb:Scan", "dynamodb:GetItem"] resources = [var.linegroups_table_arn] }
  statement { actions = ["dynamodb:PutItem", "dynamodb:GetItem"] resources = [var.systemstate_table_arn] }
  statement { actions = ["ssm:GetParameter"] resources = [var.line_access_token_param_arn] }
}

resource "aws_iam_role" "push" {
  name               = "${var.project_name}-line-push-role"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

resource "aws_iam_role_policy" "push_logs" { role = aws_iam_role.push.id policy = data.aws_iam_policy_document.logs_write.json }
resource "aws_iam_role_policy" "push_inline" { role = aws_iam_role.push.id policy = data.aws_iam_policy_document.push_inline.json }

output "webhook_role_arn" { value = aws_iam_role.webhook.arn }
output "ingest_role_arn"  { value = aws_iam_role.ingest.arn }
output "push_role_arn"    { value = aws_iam_role.push.arn }
