# One small ARM instance running the bot via docker compose.
# - No inbound ports: admin access is via AWS SSM Session Manager (no SSH keys to leak).
# - Secrets live in SSM Parameter Store (SecureString, encrypted with a dedicated KMS key);
#   the instance role can read ONLY /kronos/* and decrypt ONLY with that key.
# - Elastic IP gives a fixed egress IP to whitelist on the exchange API key.
# Approx cost (us-east-1 on-demand list prices at time of writing; check the AWS calculator): t4g.small ~$12.3/mo,
# 20 GB gp3 ~$1.6/mo, public IPv4 ~$3.7/mo, KMS key $1/mo  => ~$19/mo.
# Running `terraform apply` CREATES BILLABLE RESOURCES. Review `terraform plan` first.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.70" }
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = { project = "kronos-autotrader", managed_by = "terraform" } }
}

data "aws_ami" "al2023_arm" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-arm64"]
  }
}

resource "aws_kms_key" "kronos" {
  description             = "Kronos autotrader secrets"
  enable_key_rotation     = true
  deletion_window_in_days = 7
}

# Placeholders only - set real values with:
#   aws ssm put-parameter --name /kronos/EXCHANGE_API_KEY --type SecureString \
#     --key-id alias/kronos --value '...' --overwrite
resource "aws_kms_alias" "kronos" {
  name          = "alias/kronos"
  target_key_id = aws_kms_key.kronos.key_id
}

resource "aws_ssm_parameter" "secret" {
  for_each = toset(var.secret_names)
  name     = "/kronos/${each.key}"
  type     = "SecureString"
  key_id   = aws_kms_key.kronos.arn
  value    = "CHANGE_ME"
  lifecycle { ignore_changes = [value] } # real values are set out of band, never in TF state
}

resource "aws_iam_role" "bot" {
  name = "kronos-bot"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.bot.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "read_secrets" {
  name = "kronos-read-secrets"
  role = aws_iam_role.bot.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ssm:GetParametersByPath", "ssm:GetParameter*"],
      Resource = "arn:aws:ssm:${var.region}:${data.aws_caller_identity.me.account_id}:parameter/kronos/*" },
      { Effect = "Allow", Action = ["kms:Decrypt"], Resource = aws_kms_key.kronos.arn },
    ]
  })
}

data "aws_caller_identity" "me" {}

resource "aws_iam_instance_profile" "bot" {
  name = "kronos-bot"
  role = aws_iam_role.bot.name
}

resource "aws_security_group" "bot" {
  name        = "kronos-bot"
  description = "No inbound; outbound HTTPS only"
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 53
    to_port     = 53
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "bot" {
  ami                    = data.aws_ami.al2023_arm.id
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.bot.name
  vpc_security_group_ids = [aws_security_group.bot.id]
  user_data = templatefile("${path.module}/user_data.sh", {
    region = var.region, image = var.image, repo_url = var.repo_url, repo_ref = var.repo_ref
  })
  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }
  root_block_device {
    volume_size = 20
    volume_type = "gp3"
    encrypted   = true
  }
  tags = { Name = "kronos-bot" }
}

resource "aws_eip" "bot" {
  instance = aws_instance.bot.id
  domain   = "vpc"
}

# Billing guard-rail: email when forecast monthly spend exceeds the budget.
resource "aws_budgets_budget" "monthly" {
  name         = "kronos-monthly"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}
