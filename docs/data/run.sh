#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

TOP_DIR=$(cd ${CUR_DIR}/../..; pwd)

source ${TOP_DIR}/_env

AI_HOST=${HOST_LAN_IP:-'0.0.0.0'}
# AI_HOST=10.70.22.66

# curl -XPOST -H "Content-Type:application/json" -d@${CUR_DIR}/dogcat.json http://${AI_HOST}:8119/k12ai/framework/execute 
# curl -XPOST -H "Content-Type:application/json" -d@${CUR_DIR}/hellopyr.json http://${AI_HOST}:8119/k12ai/framework/execute 
curl -XPOST -H "Content-Type:application/json" -d@${CUR_DIR}/cr.json http://${AI_HOST}:9119/raceai/framework/inference
