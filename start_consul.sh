#!/bin/bash
#=================================================================
# date: 2019-11-18 19:32:57
# title: start_consul
# author: QRS
#=================================================================

docker run -dit \
           --restart=always \
           --name=gamma-consul \
           --volume /data/:/data \
           --network host \
           --hostname gamma-consul \
           consul agent -dev -server -bootstrap \
           -node=gamma \
           -data-dir=/data/consul \
           -datacenter=gamma -client='0.0.0.0' -ui
