#!/bin/bash

mkdir -p /data/minio/config

mkdir -p /data/minio/data

docker run -d --network host --name minio --restart=always \
     -e "MINIO_ROOT_USER=minioadmin" \
     -e "MINIO_ROOT_PASSWORD=minioadmin" \
     -v /data/minio/data:/data \
     -v /data/minio/config:/root/.minio \
     minio/minio server \
     /data --console-address ":9090" -address ":9000"

# http://82.157.36.183:9090/
