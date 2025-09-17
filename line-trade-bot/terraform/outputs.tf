output "api_endpoint" {
  value       = aws_apigatewayv2_api.http.api_endpoint
  description = "Base URL of HTTP API"
}

output "webhook_endpoint" {
  value       = "${aws_apigatewayv2_api.http.api_endpoint}/line/webhook"
  description = "LINE Webhook endpoint"
}

output "events_endpoint" {
  value       = "${aws_apigatewayv2_api.http.api_endpoint}/events"
  description = "Trade events ingestion endpoint"
}

output "trade_events_topic_arn" {
  value       = aws_sns_topic.trade_events.arn
  description = "SNS Topic ARN for trade events"
}

output "dynamodb_tables" {
  value = {
    subscribers = aws_dynamodb_table.subscribers.name
    events      = aws_dynamodb_table.events.name
    linegroups  = aws_dynamodb_table.line_groups.name
    systemstate = aws_dynamodb_table.system_state.name
  }
  description = "DynamoDB table names"
}

