#!/bin/bash

docker run -itd \
    --name redis \
    --publish 10090:6379 \
    redis --requirepass "qY3Zh4xLPZNMkaz3"
