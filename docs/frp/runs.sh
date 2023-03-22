#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

docker run -d --name frps --restart unless-stopped \
    --network host --hostname frps \
    --env FRP_SERVER_PORT=7000 \
    -v ${CUR_DIR}/frps.ini:/etc/frp/frps.ini \
    snowdreamtech/frps
    
