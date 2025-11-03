variable "project_name" { type = string }
variable "tags" { type = map(string) default = {} }

resource "aws_sns_topic" "trade_events" {
  name = "trade-events"
  tags = var.tags
}

output "topic_arn" { value = aws_sns_topic.trade_events.arn }

