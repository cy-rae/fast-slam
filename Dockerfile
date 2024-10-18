# Use the jde robotics backend image as the base image
FROM jderobot/robotics-backend:latest

# Bosch Proxy TODO
ENV http_proxy=http://rb-proxy-de.bosch.com:8080
ENV https_proxy=http://rb-proxy-de.bosch.com:8080

# Update the package lists
RUN apt-get update

# Copy the FastSLAM 2.0 module into the container
WORKDIR /workspace/code
COPY setup.py .
COPY fast_slam_2/ fast_slam_2/

# Install all necessary packages specified in requirements.txt
RUN pip install --no-cache-dir -r fast_slam_2/requirements.txt

# Install the FastSLAM 2.0 module
RUN pip install .

# Return to the root directory
WORKDIR /