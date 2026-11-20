variable "region" {
  description = "AWS region for the endpoint and its artifacts."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "Instance type behind the SageMaker endpoint. The exported graph is CPU sized."
  type        = string
  default     = "ml.m7i.large"
}

variable "min_capacity" {
  description = "Instances kept warm."
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Ceiling for target-tracking autoscaling."
  type        = number
  default     = 4
}

variable "invocations_per_instance" {
  description = "Target invocations per minute per instance that autoscaling aims for."
  type        = number
  default     = 1200
}

variable "p95_latency_ms_slo" {
  description = "Release gate SLO; the alarm fires when the endpoint misses it."
  type        = number
  default     = 250
}

variable "monthly_budget_usd" {
  description = "Budget for the whole project, with alerts at 80 and 100 percent."
  type        = number
  default     = 40
}

variable "alert_email" {
  description = "Where budget and alarm notices go. Left empty, no subscription is created."
  type        = string
  default     = ""
}

variable "image_uri" {
  description = <<-EOT
    Serving container for the endpoint. Empty means the AWS-managed ONNX Runtime image for
    the chosen region, which is what the default plan uses.
  EOT
  type        = string
  default     = ""
}
