#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-15 18:53:37

import os
import json
import argparse
import logging
import zerorpc
import socket
from flask import Flask, request, jsonify

app = Flask(__name__)

cliCV = zerorpc.Client()
cliNLP = zerorpc.Client()

__SERVERS = {}

if os.environ.get("K12AI_DEBUG"):
    LEVEL = logging.DEBUG
    app.config['DEBUG'] = True
else:
    LEVEL = logging.INFO
    app.config['DEBUG'] = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

logger = logging.getLogger(__name__)

@app.route('/k12ai/service/message', methods=['GET'])
def _rpc_service_message():
    logging.info("service request")
    try:
        reqJsonDic = json.loads(request.get_data().decode())
        name = request.args.get('server_name')
        host = request.args.get('server_host')
        port = request.args.get('server_port')
        __SERVERS[name] = 'tcp://%s:%s' % (host, port)
    except Exception as e:
        print(e)
        return "error"

    return '1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--host',
            default="0.0.0.0",
            type=str,
            dest='host',
            help="host to run k12ai service")
    parser.add_argument(
            '--port',
            default=8129,
            type=int,
            dest='port',
            help="port to run k12ai service")
    args = parser.parse_args()

    logger.info('start ai server on %s:%d', args.host, args.port)

    app.run(host=args.host, port=args.port)
