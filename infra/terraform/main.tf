module "dynamodb" {
  source = "./modules/dynamodb"
  tags   = var.tags
}

module "sns" {
  source       = "./modules/sns"
  project_name = var.project_name
  tags         = var.tags
}

module "iam" {
  source = "./modules/iam"

  project_name = var.project_name
  # tables
  subscribers_table_arn = module.dynamodb.subscribers_arn
  events_table_arn      = module.dynamodb.events_arn
  linegroups_table_arn  = module.dynamodb.linegroups_arn
  systemstate_table_arn = module.dynamodb.systemstate_arn

  # ssm parameter arns
  line_access_token_param_arn = module.iam_common.ssm_access_token_arn
  line_secret_param_arn       = module.iam_common.ssm_secret_arn
  ingest_token_param_arn      = module.iam_common.ssm_ingest_arn

  # outputs
  depends_on = [module.iam_common]
}

# helper to compute SSM parameter arns
module "iam_common" {
  source = "./modules/iam/common"

  aws_region                         = var.aws_region
  line_channel_access_token_param    = var.line_channel_access_token_param_name
  line_channel_secret_param          = var.line_channel_secret_param_name
  ingest_auth_token_param            = var.ingest_auth_token_param_name
}

module "lambda" {
  source = "./modules/lambda"

  project_name = var.project_name
  deps_layer_zip = var.deps_layer_zip

  # roles
  webhook_role_arn = module.iam.webhook_role_arn
  ingest_role_arn  = module.iam.ingest_role_arn
  push_role_arn    = module.iam.push_role_arn

  # tables names
  subscribers_table = module.dynamodb.subscribers_name
  events_table      = module.dynamodb.events_name
  linegroups_table  = module.dynamodb.linegroups_name
  systemstate_table = module.dynamodb.systemstate_name

  # ssm param names for runtime reads
  line_access_token_param = var.line_channel_access_token_param_name
  line_secret_param       = var.line_channel_secret_param_name
  ingest_token_param      = var.ingest_auth_token_param_name

  # sns
  trade_events_topic_arn = module.sns.topic_arn

  tags = var.tags
}

module "apigw" {
  source       = "./modules/apigw"
  project_name = var.project_name

  webhook_function_arn  = module.lambda.webhook_function_arn
  webhook_function_name = module.lambda.webhook_function_name
  ingest_function_arn   = module.lambda.ingest_function_arn
  ingest_function_name  = module.lambda.ingest_function_name
}

