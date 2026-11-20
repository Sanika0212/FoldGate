// AWS side of FoldGate: where a promoted version would be served.
//
//   artifacts (S3, KMS)  ->  SageMaker model  ->  endpoint with two production variants
//                                                 (champion 80%, candidate 20%)
//   CloudWatch           ->  alarms on p95 latency, 5xx and the abstention rate
//   Budgets              ->  monthly cap with alerts, because this is a portfolio project
//
// This stack is written, formatted, validated and linted in CI. It has not been applied:
// no FoldGate endpoint has ever run in an AWS account, and the README says so.

data "aws_caller_identity" "me" {}

data "aws_partition" "current" {}

locals {
  name = "foldgate"
  // Managed ONNX Runtime serving image, per region, unless one is passed in.
  image = var.image_uri != "" ? var.image_uri : "763104351884.dkr.ecr.${var.region}.amazonaws.com/onnxruntime-inference:1.19.2-cpu-py312"
}

# --------------------------------------------------------------------- artifacts

resource "aws_kms_key" "artifacts" {
  description             = "FoldGate model artifacts"
  enable_key_rotation     = true
  deletion_window_in_days = 7
}

resource "aws_kms_alias" "artifacts" {
  name          = "alias/${local.name}-artifacts"
  target_key_id = aws_kms_key.artifacts.key_id
}

resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "${local.name}-artifacts-"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.artifacts.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# ------------------------------------------------------------------------- iam

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "endpoint" {
  name               = "${local.name}-endpoint"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

data "aws_iam_policy_document" "endpoint" {
  statement {
    sid       = "ReadArtifacts"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
  }

  statement {
    sid       = "DecryptArtifacts"
    actions   = ["kms:Decrypt", "kms:DescribeKey"]
    resources = [aws_kms_key.artifacts.arn]
  }

  statement {
    sid       = "PublishMetricsAndLogs"
    actions   = ["cloudwatch:PutMetricData", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "endpoint" {
  name   = "${local.name}-endpoint"
  role   = aws_iam_role.endpoint.id
  policy = data.aws_iam_policy_document.endpoint.json
}

# -------------------------------------------------------------------- endpoint

resource "aws_sagemaker_model" "champion" {
  name               = "${local.name}-champion"
  execution_role_arn = aws_iam_role.endpoint.arn

  primary_container {
    image          = local.image
    model_data_url = "s3://${aws_s3_bucket.artifacts.id}/versions/champion/model.tar.gz"

    environment = {
      FOLDGATE_ALPHA         = "0.1"
      FOLDGATE_ABSTAIN_WIDTH = "2.0"
    }
  }
}

resource "aws_sagemaker_model" "candidate" {
  name               = "${local.name}-candidate"
  execution_role_arn = aws_iam_role.endpoint.arn

  primary_container {
    image          = local.image
    model_data_url = "s3://${aws_s3_bucket.artifacts.id}/versions/candidate/model.tar.gz"

    environment = {
      FOLDGATE_ALPHA         = "0.1"
      FOLDGATE_ABSTAIN_WIDTH = "2.0"
    }
  }
}

# Two variants, so the A/B split the router does locally is also how it would run here.
resource "aws_sagemaker_endpoint_configuration" "this" {
  name        = "${local.name}-config"
  kms_key_arn = aws_kms_key.artifacts.arn

  production_variants {
    variant_name           = "champion"
    model_name             = aws_sagemaker_model.champion.name
    initial_instance_count = var.min_capacity
    instance_type          = var.instance_type
    initial_variant_weight = 80
  }

  production_variants {
    variant_name           = "candidate"
    model_name             = aws_sagemaker_model.candidate.name
    initial_instance_count = 1
    instance_type          = var.instance_type
    initial_variant_weight = 20
  }

  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = 20
    destination_s3_uri          = "s3://${aws_s3_bucket.artifacts.id}/capture/"
    kms_key_id                  = aws_kms_key.artifacts.arn

    capture_options {
      capture_mode = "Input"
    }

    capture_options {
      capture_mode = "Output"
    }
  }
}

resource "aws_sagemaker_endpoint" "this" {
  name                 = local.name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.this.name
}

resource "aws_appautoscaling_target" "champion" {
  service_namespace  = "sagemaker"
  resource_id        = "endpoint/${aws_sagemaker_endpoint.this.name}/variant/champion"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  min_capacity       = var.min_capacity
  max_capacity       = var.max_capacity
}

resource "aws_appautoscaling_policy" "champion" {
  name               = "${local.name}-invocations"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.champion.service_namespace
  resource_id        = aws_appautoscaling_target.champion.resource_id
  scalable_dimension = aws_appautoscaling_target.champion.scalable_dimension

  target_tracking_scaling_policy_configuration {
    target_value       = var.invocations_per_instance
    scale_in_cooldown  = 300
    scale_out_cooldown = 60

    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
  }
}

# ---------------------------------------------------------------------- alarms

resource "aws_sns_topic" "alerts" {
  name              = "${local.name}-alerts"
  kms_master_key_id = aws_kms_key.artifacts.id
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "latency" {
  alarm_name          = "${local.name}-p95-latency"
  alarm_description   = "The endpoint is missing the latency SLO the release gate checks."
  namespace           = "AWS/SageMaker"
  metric_name         = "ModelLatency"
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 2
  threshold           = var.p95_latency_ms_slo * 1000 # the metric is in microseconds
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.this.name
    VariantName  = "champion"
  }
}

resource "aws_cloudwatch_metric_alarm" "errors" {
  alarm_name          = "${local.name}-5xx"
  alarm_description   = "Model errors returned to callers."
  namespace           = "AWS/SageMaker"
  metric_name         = "Invocation5XXErrors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.this.name
  }
}

# The serving container publishes this one: the share of variants the model abstained on.
# A jump means the inputs drifted away from what the model was calibrated on.
resource "aws_cloudwatch_metric_alarm" "abstention" {
  alarm_name          = "${local.name}-abstention-rate"
  alarm_description   = "Abstention above a fifth of variants suggests the inputs have shifted."
  namespace           = "FoldGate"
  metric_name         = "AbstentionRate"
  statistic           = "Average"
  period              = 900
  evaluation_periods  = 2
  threshold           = 0.2
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.this.name
  }
}

resource "aws_budgets_budget" "monthly" {
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:project$foldgate"]
  }

  dynamic "notification" {
    for_each = var.alert_email == "" ? [] : [80, 100]

    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }
}
