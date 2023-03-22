#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

TOP_DIR=$(cd ../..; pwd)

source ${TOP_DIR}/_env

AI_HOST=${HOST_LAN_IP:-'172.21.0.13'}

# curl -XPOST -H "Content-Type:application/json" -d@dogcat.json http://${AI_HOST}:8119/k12ai/framework/execute 
curl -XPOST -H "Content-Type:application/json" -d@hellopyr.json http://${AI_HOST}:8119/k12ai/framework/execute 
