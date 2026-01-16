output "secrets" {
  value = jsondecode(data.aws_secretsmanager_secret_version.current.secret_string)
}