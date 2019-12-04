#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-15 18:53:37

import os, time, json
import argparse
import logging
import zerorpc
import socket
import consul
import subprocess
from flask import Flask, request, jsonify
from threading import Thread

from k12ai_errmsg import hzcsk12_error_message as _err_msg

if os.environ.get("K12AI_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app_quit = False

def _get_host_name():
    val = os.environ.get('HOST_NAME', None)
    if val:
        return val
    else:
        return socket.gethostname()

def _get_host_ip():
    val = os.environ.get('HOST_ADDR', None)
    if val:
        return val
    else:
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
                    check=consul.Check.http('http://{}:{}/status'.format(host, port),
                        interval='10s', timeout='5s', deregister='10s'))
            break
        except Exception as err:
            logger.error("consul agent service register err", err)
            time.sleep(3)

class RPCServiceAgent(object):
    def __init__(self, addr, port):
        self._addr = addr
        self._port = port

    def __getattr__(self, method):
        def __wrapper(*args, **kwargs):
            try:
                client = zerorpc.Client(timeout=15)
                client.connect('tcp://{}:{}'.format(self._addr, self._port))
                result = client(method, *args, **kwargs)
                client.close()
                return result
            except Exception as err:
                raise err
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
    return "Success"

### Platform resource manager
platform_service_name = 'k12platform'

@app.route('/k12ai/platform/stats', methods=['POST'])
def _platform_stats():
    logger.info('call _platform_stats')
    try:
        reqjson = json.loads(request.get_data().decode())
        username = reqjson['username']
        password = reqjson['password']
        isasync = reqjson.get('async', False)
        query = reqjson.get('query', None)
    except Exception as err:
        return json.dumps(_err_msg(100101, exc=True))

    # TODO check username and password

    agent = _get_service_by_name(platform_service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{platform_service_name}'))

    try:
        code, msg = agent.stats(query, isasync)
        return json.dumps(_err_msg(100202 if code < 0 else 100200, msg))
    except Exception as err:
        return json.dumps(_err_msg(100202, exc=True))

@app.route('/k12ai/platform/control', methods=['POST'])
def _platform_control():
    logger.info('call _platform_control')
    try:
        reqjson = json.loads(request.get_data().decode())
        username = reqjson['username']
        password = reqjson['password']
        op = reqjson['op']
        isasync = reqjson.get('async', False)
        params = reqjson.get('params', None)
    except Exception as err:
        return json.dumps(_err_msg(100101, exc=True))

    # TODO check username and password

    agent = _get_service_by_name(platform_service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{platform_service_name}'))
    try:
        code, msg = agent.control(op, params, isasync)
        return json.dumps(_err_msg(100202 if code < 0 else 100200, msg))
    except Exception as err:
        return json.dumps(_err_msg(100202, exc=True))

### GPU Framework for train/evaluate/predict
@app.route('/k12ai/framework/train', methods=['POST'])
def _framework_train():
    logger.info('call _framework_train')
    try:
        reqjson = json.loads(request.get_data().decode())
        user = reqjson['user']
        op = reqjson['op']
        if op not in ('train.start', 'train.stop'):
            return json.dumps(_err_msg(100102, f'not support op:{op}'))
        service_name = reqjson['service_name']
        service_uuid = reqjson['service_uuid']
        service_params = reqjson.get('service_params', None)
    except Exception as err:
        return json.dumps(_err_msg(100101, exc=True))

    agent = _get_service_by_name(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))
    try:
        code, msg = agent.train(op, user, service_uuid, service_params)
        return json.dumps(_err_msg(100202 if code < 0 else 100200, msg))
    except Exception as err:
        return json.dumps(_err_msg(100202, exc=True))

@app.route('/k12ai/framework/evaluate', methods=['POST'])
def _framework_evaluate():
    logger.info('call _framework_evaluate')
    try:
        reqjson = json.loads(request.get_data().decode())
        user = reqjson['user']
        op = reqjson['op']
        if op not in ('evaluate.start', 'evaluate.stop'):
            return json.dumps(_err_msg(100102, f'not support op:{op}'))
        service_name = reqjson['service_name']
        service_uuid = reqjson['service_uuid']
        service_params = reqjson.get('service_params', None)
    except Exception as err:
        return json.dumps(_err_msg(100101, exc=True))

    agent = _get_service_by_name(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))
    try:
        code, msg = agent.evaluate(op, user, service_uuid, service_params)
        return json.dumps(_err_msg(100202 if code < 0 else 100200, msg))
    except Exception as err:
        return json.dumps(_err_msg(100202, exc=True))

@app.route('/k12ai/framework/predict', methods=['POST'])
def _framework_predict():
    logger.info('call _framework_predict')
    try:
        reqjson = json.loads(request.get_data().decode())
        user = reqjson['user']
        op = reqjson['op']
        if op not in ('predict.start', 'predict.stop'):
            return json.dumps(_err_msg(100102, f'not support op:{op}'))
        service_name = reqjson['service_name']
        service_uuid = reqjson['service_uuid']
        service_params = reqjson.get('service_params', None)
    except Exception as err:
        return json.dumps(_err_msg(100101, exc=True))

    agent = _get_service_by_name(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))
    try:
        code, msg = agent.predict(op, user, service_uuid, service_params)
        return json.dumps(_err_msg(100202 if code else 100200, msg))
    except Exception as err:
        return json.dumps(_err_msg(100202, exc=True))

@app.route('/k12ai/private/message', methods=['POST'])
def _framework_message():
    logger.info('call _framework_message')
    try:
        reqjson = request.json
    except Exception as err:
        logger.info(err)
        return "error"
    logger.info(reqjson)
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
            default=8119,
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
