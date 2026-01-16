variable "instance_ami" { type = string }
variable "instance_type" { type = string }
variable "subnet_id" { type = string }
variable "ssh_public_key" { type = string }
variable "ssh_key_name" { type = string }
variable "user_data" { type = string }
variable "vpc_id" { type = string }
variable "security_group_name" { type = string }