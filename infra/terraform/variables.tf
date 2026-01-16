variable "availability_zone" {
  description = "Availability zone of resources"
  type        = string
}
variable "instance_ami" {
  description = "ID of the AMI used"
  type        = string
}
variable "instance_type" {
  description = "Type of the instance"
  type        = string
}
variable "docker_image_tag" {
  description = "Docker image tag for the application"
  type        = string
}
variable "domain" {
  description = "Domain name for the application"
  type        = string
}
variable "app_url" {
  description = "Application URL"
  type        = string
}