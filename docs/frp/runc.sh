#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
TOP_DIR=$(cd ../..; pwd)

source ${TOP_DIR}/_env

docker run -d --name frpc --restart unless-stopped \
    --network host --hostname frpc \
    --env FRP_SERVER_ADDR=${FRP_SERVER_ADDR} \
    --env FRP_SERVER_PORT=7000 \
    --env FRP_LOCAL_IP=${HOST_LAN_IP} \
    -v ${CUR_DIR}/frpc.ini:/etc/frp/frpc.ini \
    snowdreamtech/frpc
