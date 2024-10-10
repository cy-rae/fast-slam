# Verwende das jderobot/robotics-backend:latest Image als Basis
FROM jderobot/robotics-backend:latest

# Bosch Proxy
ENV http_proxy=http://rb-proxy-de.bosch.com:8080
ENV https_proxy=http://rb-proxy-de.bosch.com:8080

# Install dependencies
RUN apt-get update

# Copy the fast_slam_2 module into the container
WORKDIR /workspace/code
COPY setup.py .
COPY fast_slam_2/ fast_slam_2/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r fast_slam_2/requirements.txt

# Install the fast_slam_2 module
RUN pip install .

# Return to the root directory
WORKDIR /