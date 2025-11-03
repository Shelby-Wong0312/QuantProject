output "api_endpoint" {
  value       = module.apigw.api_endpoint
  description = "Base URL of HTTP API"
}

output "webhook_endpoint" {
  value       = module.apigw.webhook_endpoint
  description = "LINE Webhook endpoint"
}

output "events_endpoint" {
  value       = module.apigw.events_endpoint
  description = "Trade events ingestion endpoint"
}

output "trade_events_topic_arn" {
  value       = module.sns.topic_arn
  description = "SNS Topic ARN for trade events"
}

output "dynamodb_tables" {
  value = {
    subscribers = module.dynamodb.subscribers_name
    events      = module.dynamodb.events_name
    linegroups  = module.dynamodb.linegroups_name
    systemstate = module.dynamodb.systemstate_name
  }
  description = "DynamoDB table names"
}

