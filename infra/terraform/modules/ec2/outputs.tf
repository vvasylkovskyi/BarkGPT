output "instance_id" { value = aws_instance.my_app.id }
output "public_ip"   { value = aws_eip.my_app.public_ip }