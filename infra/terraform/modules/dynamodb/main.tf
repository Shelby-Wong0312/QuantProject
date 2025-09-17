variable "tags" { type = map(string) default = {} }

resource "aws_dynamodb_table" "subscribers" {
  name         = "trade-events-subscribers"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "recipientId"
  attribute { name = "recipientId" type = "S" }
  sse_specification { enabled = true }
  tags = var.tags
}

resource "aws_dynamodb_table" "events" {
  name         = "TradeEvents"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "ts"
  attribute { name = "ts" type = "S" }
  attribute { name = "bucket" type = "S" }
  global_secondary_index {
    name            = "recent-index"
    hash_key        = "bucket"
    range_key       = "ts"
    projection_type = "ALL"
  }
  sse_specification { enabled = true }
  tags = var.tags
}

resource "aws_dynamodb_table" "line_groups" {
  name         = "LineGroups"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "targetId"
  attribute { name = "targetId" type = "S" }
  sse_specification { enabled = true }
  tags = var.tags
}

resource "aws_dynamodb_table" "system_state" {
  name         = "SystemState"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  attribute { name = "pk" type = "S" }
  sse_specification { enabled = true }
  tags = var.tags
}

output "subscribers_name" { value = aws_dynamodb_table.subscribers.name }
output "events_name"      { value = aws_dynamodb_table.events.name }
output "linegroups_name"  { value = aws_dynamodb_table.line_groups.name }
output "systemstate_name" { value = aws_dynamodb_table.system_state.name }

output "subscribers_arn" { value = aws_dynamodb_table.subscribers.arn }
output "events_arn"      { value = aws_dynamodb_table.events.arn }
output "linegroups_arn"  { value = aws_dynamodb_table.line_groups.arn }
output "systemstate_arn" { value = aws_dynamodb_table.system_state.arn }
