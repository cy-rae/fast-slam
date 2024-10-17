#!/bin/bash

# Install the requirements for the map main
echo "Install the requirements for the map main..."
pip install -r requirements.txt

# Create shared folder if it does not exist
SHARED_DIR="C:/shared"  # Change this to the path of the shared folder on your host machine

if [ ! -d "$SHARED_DIR" ]; then
    echo "Create shared folder $SHARED_DIR..."
    mkdir -p "$SHARED_DIR"
else
    echo "The folder $SHARED_DIR already exists."
fi

# Build the Docker image
IMAGE_NAME="fast-slam"
echo "Build the Docker image $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# Start the docker container with the mounted shared folder
echo "Start the Docker container with the mounted shared folder..."
docker run -v $SHARED_DIR:/workspace/shared -d -p 7164:7164 -p 6080:6080 -p 1108:1108 -p 7163:7163 $IMAGE_NAME

echo "Docker-Container wurde gestartet."