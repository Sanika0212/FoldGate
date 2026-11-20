output "endpoint_name" {
  description = "SageMaker endpoint that would serve the champion and the candidate."
  value       = aws_sagemaker_endpoint.this.name
}

output "artifacts_bucket" {
  description = "Where model.tar.gz per version and the captured traffic would live."
  value       = aws_s3_bucket.artifacts.id
}

output "alerts_topic_arn" {
  description = "SNS topic carrying the latency, error and abstention alarms."
  value       = aws_sns_topic.alerts.arn
}

output "account_id" {
  description = "Account the stack would be applied to."
  value       = data.aws_caller_identity.me.account_id
}

output "partition" {
  description = "AWS partition, so ARNs built from this stack stay correct outside commercial regions."
  value       = data.aws_partition.current.partition
}
