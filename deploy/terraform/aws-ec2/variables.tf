variable "region" {
  description = "Pick one close to your exchange and allowed for your residency (e.g. eu-west-3 Paris, ap-south-1 Mumbai, us-east-1)."
  type        = string
  default     = "eu-west-3"
}

variable "instance_type" {
  description = "t4g.small = 2 vCPU / 2 GB ARM. Enough for Kronos-small on CPU at 4h/1d bars."
  type        = string
  default     = "t4g.small"
}

variable "image" {
  description = "Container image (multi-arch) built by .github/workflows/deploy.yml"
  type        = string
}

variable "repo_url" {
  description = "Your fork (for deploy/ configs)"
  type        = string
}

variable "repo_ref" {
  type    = string
  default = "main"
}

variable "secret_names" {
  type = list(string)
  default = [
    "EXCHANGE_API_KEY", "EXCHANGE_API_SECRET", "SLACK_WEBHOOK_URL",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "NTFY_TOPIC", "KRONOS_LIVE_CONFIRM", "GHCR_TOKEN",
  ]
}

variable "monthly_budget_usd" {
  type    = string
  default = "30"
}

variable "alert_email" {
  type = string
}
