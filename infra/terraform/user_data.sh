#!/bin/bash
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USERNAME

sudo docker run -d --name bark-gpt-api -p 80:80 vvasylkovskyi1/vvasylkovskyi-bark-gpt-api:${docker_image_tag}