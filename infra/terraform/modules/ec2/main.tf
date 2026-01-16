resource "aws_key_pair" "ssh-key" {
  key_name   = var.ssh_key_name
  public_key = var.ssh_public_key
}

resource "aws_security_group" "my_app" {
  name   = var.security_group_name
  vpc_id = var.vpc_id

  ingress {
    cidr_blocks = ["0.0.0.0/0"]
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
  }
  
  ingress {
    cidr_blocks = ["0.0.0.0/0"]
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "my_app" {
  ami                         = var.instance_ami
  instance_type               = var.instance_type
  security_groups             = [aws_security_group.my_app.id]
  associate_public_ip_address = true
  subnet_id                   = var.subnet_id
  key_name                    = aws_key_pair.ssh-key.key_name
  user_data                   = var.user_data
}

resource "aws_eip" "my_app" {
  instance = aws_instance.my_app.id
  domain   = "vpc"
}