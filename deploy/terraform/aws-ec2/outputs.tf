output "egress_ip" {
  description = "Whitelist this IP on the exchange API key"
  value       = aws_eip.bot.public_ip
}

output "instance_id" {
  description = "Connect with: aws ssm start-session --target <id>"
  value       = aws_instance.bot.id
}
