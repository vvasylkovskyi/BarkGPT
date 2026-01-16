provider "aws" {
  region                   = "us-east-1"
  shared_credentials_files = ["./.aws-credentials"]
  profile                  = "terraform"
}

module "network" {
  source            = "./modules/network"
  vpc_cidr          = "10.0.0.0/16"
  subnet_cidr       = "10.0.1.0/24"
  availability_zone = var.availability_zone
}

module "ec2_first_instance" {
  source              = "./modules/ec2"
  instance_ami        = var.instance_ami
  instance_type       = var.instance_type
  vpc_id              = module.network.vpc_id
  subnet_id           = module.network.public_subnet_ids[0]
  ssh_public_key      = module.secrets.secrets.ssh_public_key
  ssh_key_name        = "ec2-key-first-instance"
  security_group_name = "first-ec2-instance"
  user_data = templatefile("${path.module}/user_data.sh", { docker_image_tag = var.docker_image_tag })
}

data "aws_route53_zone" "main" {
  name         = var.domain
}

module "ssl_acm" {
  source              = "./modules/acm"
  aws_route53_zone_id = data.aws_route53_zone.main.zone_id
  domain_name         = var.domain
  app_url             = var.app_url
}

module "api_gateway" {
  source              = "./modules/api-gateway"
  api_name            = "my-api"
  domain_name         = var.app_url
  acm_certificate_arn = module.ssl_acm.aws_acm_certificate_arn
  ec2_public_url      = "http://${module.ec2_first_instance.public_ip}:80"
}

module "aws_route53_record" {
  source       = "./modules/dns"
  main_zone_id = data.aws_route53_zone.main.zone_id
  target_domain_name = module.api_gateway.aws_apigatewayv2_domain_name
  hosted_zone_id    = module.api_gateway.aws_apigatewayv2_hosted_zone_id
  dns_record_url = var.app_url
}

module "secrets" {
  source           = "./modules/secrets"
  credentials_name = "my_app/v1/credentials"
}
