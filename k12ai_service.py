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
import consul
import time
from flask import Flask, request, jsonify
from threading import Thread

if os.environ.get("K12AI_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

# TODO
LEVEL = logging.DEBUG

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app_quit = False

def _get_host_name():
    return socket.gethostname()

def _get_host_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

app_host_name = _get_host_name()
app_host_ip = _get_host_ip()

consul_addr = None
consul_port = None

def _delay_do_consul(host, port):
    time.sleep(3)
    while not app_quit:
        try:
            client = consul.Consul(host=consul_addr, port=consul_port)
            client.agent.service.register(
                    name='{}-k12ai'.format(app_host_name),
                    address=host, port=port, tags=('AI', 'ML'),
                    check=consul.Check.http('http://{}:{}/status'.format(host, port), interval='5s'))
            break
        except Exception as err:
            logger.info("consul agent service register err", err)
            time.sleep(3)

class RPCServiceAgent(object):
    def __init__(self, addr, port):
        self._addr = addr
        self._port = port

    def __getattr__(self, method):
        def __wrapper(*args, **kwargs):
            try:
                client = zerorpc.Client(timeout=3)
                client.connect('tcp://{}:{}'.format(self._addr, self._port))
                result = client(method, *args, **kwargs)
                client.close()
                return result
            except Exception as err:
                logger.info(err)
                return None
        return __wrapper

def _get_service_by_name(name):
    client = consul.Consul(consul_addr, port=consul_port)
    service = client.agent.services().get('%s-%s' % (app_host_name, name))
    if not service:
        return None
    else:
        return RPCServiceAgent(service['Address'], service['Port'])

### Consul check the flask service health
@app.route('/status', methods=['GET'])
def _consul_check_status():
    return "1"

### GPU Framework for train/evaluate/predict
@app.route('/k12ai/framework/train', methods=['POST'])
def _framework_train():
    logger.info('call _framework_train')
    try:
        reqjson = request.json
        service_name = reqjson['service_name']
        cmd = reqjson['service_cmd']
    except Exception as err:
        logger.info(err)
        return "error"

    if cmd == "stop":
        return 0;

    agent = _get_service_by_name(service_name)
    agent.train('aaa')

    return "1"

@app.route('/k12ai/framework/evaluate', methods=['POST'])
def _framework_evaluate():
    logger.info('call _framework_evaluate')
    try:
        reqjson = json.loads(request.get_data().decode())
    except Exception as err:
        logger.info(err)
        return "error"

    return "1"

@app.route('/k12ai/framework/predict', methods=['POST'])
def _framework_predict():
    logger.info('call _framework_predict')
    try:
        reqjson = json.loads(request.get_data().decode())
    except Exception as err:
        logger.info(err)
        return "error"

    return "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--host',
            default=None,
            type=str,
            dest='host',
            help="host to run k12ai service")
    parser.add_argument(
            '--port',
            default=8129,
            type=int,
            dest='port',
            help="port to run k12ai service")
    parser.add_argument(
            '--consul_addr',
            default=None,
            type=str,
            dest='consul_addr',
            help="consul address")
    parser.add_argument(
            '--consul_port',
            default=8500,
            type=int,
            dest='consul_port',
            help="consul port")
    args = parser.parse_args()

    host = args.host if args.host else app_host_ip

    consul_addr = args.consul_addr if args.consul_addr else app_host_ip
    consul_port = args.consul_port

    logger.info('start ai server on %s:%d', host, args.port)
    logger.info('start consul server on %s:%d', consul_addr, consul_port)

    thread = Thread(target=_delay_do_consul, args=(host, args.port))
    thread.start()

    try:
        app.run(host=host, port=args.port)
    finally:
        app_quit = True
        thread.join()
