# Verwende das jderobot/robotics-backend:latest Image als Basis
FROM jderobot/robotics-backend:latest

# Bosch Proxy
ENV http_proxy=http://rb-proxy-de.bosch.com:8080
ENV https_proxy=http://rb-proxy-de.bosch.com:8080

# Install NGINX and other dependencies
RUN apt-get update && apt-get install -y nginx

# Copy the custom Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Erstelle ein Verzeichnis für statische Dateien
RUN mkdir -p /usr/share/nginx/html/images

# Kopiere das Originalbild und die HTML-Datei in das Verzeichnis von NGINX
COPY index.html /usr/share/nginx/html/index.html

# Expose default Nginx server port 80
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]
