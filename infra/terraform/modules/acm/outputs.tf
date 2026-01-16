output "aws_acm_certificate_arn" {
  value = aws_acm_certificate_validation.cert.certificate_arn
}

output "aws_acm_certificate_cert" {
  value = aws_acm_certificate.cert
}
