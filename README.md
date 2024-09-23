# Start robot academy with ssh access

```bash
docker build -t fast-slam .
```

```bash
docker run -d -p 8080:80 -p 7164:7164 -p 6080:6080 -p 1108:1108 -p 7163:7163 fast-slam
```

## Start Nginx
If the nginx server is not running, you can start it with the following command:
1. Replace the whole content of the file `/etc/nginx/nginx.conf` with the content of the local nginx.conf. 
The format might be wrong. 
```bash
nano /etc/nginx/nginx.conf
```

2. Check if the configuration is correct:
```bash
nginx -t
```

3. Start the server:
```bash
nginx
```












# Old
```
docker build -t jderobot/robotics-backend-ssh .
```

```
docker run --rm -it -p 2222:22 -p 7164:7164 -p 6080:6080 -p 1108:1108 -p 7163:7163 jderobot/robotics-backend-ssh
```

```
docker run -d -p 2222:22 -p 7164:7164 -p 6080:6080 -p 1108:1108 -p 7163:7163 jderobot/robotics-backend-ssh
```

```
curl -fSL --output /root/.cache/JetBrains/RemoteDev/dist/e8ce6a2ea8e4d_ideaIU-243.15521.24.tar.gz https://download.jetbrains.com/idea/ideaIU-243.15521.24.tar.gz
```

```
ls -lh /root/
```

```
tar -xzf /root/ideaIU-243.15521.24.tar.gz -C /root/
```

```
ls /root/idea-IU-243.15521.24/
```