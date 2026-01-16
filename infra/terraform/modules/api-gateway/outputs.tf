output "aws_apigatewayv2_domain_name" {
  value       = aws_apigatewayv2_domain_name.custom.domain_name_configuration[0].target_domain_name
  description = "The custom domain name for the API Gateway"
}

output "aws_apigatewayv2_hosted_zone_id" {
  value       = aws_apigatewayv2_domain_name.custom.domain_name_configuration[0].hosted_zone_id
  description = "The hosted zone ID for the API Gateway custom domain"
}