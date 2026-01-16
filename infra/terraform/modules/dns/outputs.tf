output "www_dns_record" {
  value       = aws_route53_record.www.fqdn
  description = "The FQDN of the www Route53 record"
}