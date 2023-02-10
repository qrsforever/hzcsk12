#!/bin/bash

AI_HOST=172.21.0.13
# AI_HOST=172.21.0.16

# curl -XPOST -H "Content-Type:application/json" -d@dogcat.json http://${AI_HOST}:8119/k12ai/framework/execute 

curl -XPOST -H "Content-Type:application/json" -d@hellopyr.json http://${AI_HOST}:8119/k12ai/framework/execute 
