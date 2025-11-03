variable "project_name" { type = string }
variable "webhook_function_arn" { type = string }
variable "webhook_function_name" { type = string }
variable "ingest_function_arn" { type = string }
variable "ingest_function_name" { type = string }

resource "aws_apigatewayv2_api" "http" {
  name          = "${var.project_name}-api"
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
  integration_uri        = var.webhook_function_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_integration" "ingest" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = var.ingest_function_arn
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
  function_name = var.webhook_function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*/line/webhook"
}

resource "aws_lambda_permission" "apigw_ingest" {
  statement_id  = "AllowAPIGatewayInvokeIngest"
  action        = "lambda:InvokeFunction"
  function_name = var.ingest_function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*/events"
}

output "api_endpoint"     { value = aws_apigatewayv2_api.http.api_endpoint }
output "webhook_endpoint" { value = "${aws_apigatewayv2_api.http.api_endpoint}/line/webhook" }
output "events_endpoint"  { value = "${aws_apigatewayv2_api.http.api_endpoint}/events" }

