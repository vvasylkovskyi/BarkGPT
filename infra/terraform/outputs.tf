output "ec2_first_instance_ip_address" {
  value       = module.ec2_first_instance.public_ip
  description = "The Elastic IP address allocated to the first EC2 instance."
}
