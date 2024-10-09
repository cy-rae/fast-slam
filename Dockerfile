# Verwende das jderobot/robotics-backend:latest Image als Basis
FROM jderobot/robotics-backend:latest

# Bosch Proxy
ENV http_proxy=http://rb-proxy-de.bosch.com:8080
ENV https_proxy=http://rb-proxy-de.bosch.com:8080

# Install dependencies
RUN apt-get update && pip install scikit-learn

# Copy the fast_slam_2 module into the container
WORKDIR /workspace/code
COPY setup.py .
COPY fast_slam_2/ fast_slam_2/
RUN pip install .

WORKDIR /