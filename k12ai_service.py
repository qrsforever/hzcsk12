#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-15 18:53:37

import os, time, json
import _jsonnet
import argparse
import logging
import zerorpc
import consul
import redis
from flask import Flask, request
from flask_cors import CORS
from threading import Thread

from k12ai_errmsg import k12ai_error_message as _err_msg
from k12ai_utils import k12ai_get_hostname as _get_hostname
from k12ai_utils import k12ai_get_hostip as _get_hostip

if os.environ.get("K12AI_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)

app_quit = False
app_host_name = _get_hostname()
app_host_ip = _get_hostip()

consul_addr = None
consul_port = None

g_redis = None

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

### Platform

@app.route('/k12ai/platform/stats', methods=['POST'])
def _platform_stats():
    logger.info('call _platform_stats')
    try:
        reqjson = json.loads(request.get_data().decode())
        user = reqjson['user']
        op = reqjson['op']
        if op not in ('query'):
            return json.dumps(_err_msg(100102, f'not support op:{op}'))
        service_name = reqjson['service_name']
        service_uuid = reqjson['service_uuid']
        service_params = reqjson.get('service_params', None)
        isasync = reqjson.get('async', False)
    except Exception:
        return json.dumps(_err_msg(100101, exc=True))

    # TODO check username and password
    # logger.info('user: %s, passwd: %s' % (username, password))

    agent = _get_service_by_name(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))

    try:
        code, msg = agent.stats(op, user, service_uuid, service_params, isasync)
        return json.dumps(_err_msg(100202 if code < 0 else 100000, msg))
    except Exception:
        return json.dumps(_err_msg(100202, exc=True))


@app.route('/k12ai/platform/control', methods=['POST'])
def _platform_control():
    logger.info('call _platform_control')
    try:
        reqjson = json.loads(request.get_data().decode())
        user = reqjson['user']
        op = reqjson['op']
        if op not in ('container.stop'):
            return json.dumps(_err_msg(100102, f'not support op:{op}'))
        service_name = reqjson['service_name']
        service_uuid = reqjson['service_uuid']
        service_params = reqjson.get('service_params', None)
        isasync = reqjson.get('async', False)
    except Exception:
        return json.dumps(_err_msg(100101, exc=True))

    # TODO check username and password
    # logger.info('user: %s, passwd: %s' % (username, password))

    agent = _get_service_by_name(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))
    try:
        code, msg = agent.control(op, user, service_uuid, service_params, isasync)
        return json.dumps(_err_msg(100202 if code < 0 else 100000, msg))
    except Exception:
        return json.dumps(_err_msg(100202, exc=True))


### Framework

@app.route('/k12ai/framework/schema', methods=['POST'])
def _framework_schema():
    logger.info('call _framework_schema')
    try:
        reqjson = json.loads(request.get_data().decode())
        file = reqjson.get('file', None)
        if file: # TODO for test
            topdir = '/home/lidong/workspace/codes/hzcsai_com/'
            basic_json = {}
            if file in ('k12ai_basic_type.jsonnet', 
                    'k12ai_complex_type.jsonnet',
                    'k12ai_layout_type.jsonnet',
                    'k12ai_all_type.jsonnet'):
                schema_dir = os.path.join(topdir, 'hzcsnote', 'k12libs', 'templates', 'schema')
                basic_file = os.path.join(schema_dir, file) 
                if not os.path.exists(basic_file):
                    return json.dumps(_err_msg(100102, f'{file} is not exist'))
                basic_json = _jsonnet.evaluate_file(basic_file)
            elif file in ('k12ai_nlp.jsonnet'):
                schema_dir = os.path.join(topdir, 'hzcsk12', 'nlp', 'app', 'templates', 'schema')
                basic_file = os.path.join(schema_dir, file) 
                if not os.path.exists(basic_file):
                    return json.dumps(_err_msg(100102, f'{file} is not exist'))
                basic_json = _jsonnet.evaluate_file(basic_file, ext_vars={
                    'task': 'sentiment_analysis',
                    'dataset_path': '/data/datasets/nlp',
                    'dataset_name': 'sst'})
            elif file in ('k12ai_cv.jsonnet'):
                schema_dir = os.path.join(topdir, 'hzcsk12', 'cv', 'app', 'templates', 'schema')
                basic_file = os.path.join(schema_dir, file) 
                if not os.path.exists(basic_file):
                    return json.dumps(_err_msg(100102, f'{file} is not exist'))
                basic_json = _jsonnet.evaluate_file(basic_file, ext_vars={
                    'task': 'cls',
                    'dataset_name': 'mnist',
                    'dataset_root': '/datasets',
                    'checkpt_root': '/cache',
                    'pretrained_path': '/pretrained',
                    })
            return basic_json
        else:
            service_name = reqjson['service_name']
            service_task = reqjson['service_task']
            dataset_name = reqjson['dataset_name']
            network_type = reqjson.get('network_type', 'base_model')
    except json.decoder.JSONDecodeError:
        return json.dumps(_err_msg(100103, request.get_data().decode()))
    except Exception:
        return json.dumps(_err_msg(100101, reqjson, exc=True))

    agent = _get_service_by_name(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))
    try:
        code, msg = agent.schema(service_task, network_type, dataset_name)
        return json.dumps(_err_msg(code, msg))
    except Exception:
        return json.dumps(_err_msg(100207, exc=True))

@app.route('/k12ai/framework/execute', methods=['POST'])
def _framework_execute():
    logger.info('call _framework_project')
    try:
        reqjson = json.loads(request.get_data().decode())
        user = reqjson['user']
        op = reqjson['op']
        if op not in ('train.start', 'train.stop',
                'evaluate.start', 'evaluate.stop',
                'predict.start', 'predict.stop'):
            return json.dumps(_err_msg(100102, f'not support op:{op}'))
        service_name = reqjson['service_name']
        service_uuid = reqjson['service_uuid']
        service_params = reqjson.get('service_params', None)
    except Exception:
        return json.dumps(_err_msg(100101, exc=True))

    agent = _get_service_by_name(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))
    try:
        code, msg = agent.execute(op, user, service_uuid, service_params)
        return json.dumps(_err_msg(code, msg))
    except Exception:
        return json.dumps(_err_msg(100202, exc=True))


@app.route('/k12ai/private/message', methods=['POST', 'GET'])
def _framework_message():
    logger.info('call _framework_message')
    try:
        if g_redis:
            msgtype = request.args.get("type", default='unknown')
            g_redis.lpush('k12ai.{}'.format(msgtype), request.get_data().decode())
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
            default=8119,
            type=int,
            dest='port',
            help="port to run k12ai service")
    parser.add_argument(
            '--redis_addr',
            default=None,
            type=str,
            dest='redis_addr',
            help="redis address")
    parser.add_argument(
            '--redis_port',
            default=10090,
            type=int,
            dest='redis_port',
            help="redis port")
    parser.add_argument(
            '--redis_passwd',
            default='123456',
            type=str,
            dest='redis_passwd',
            help="redis passwd")
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

    redis_addr = args.redis_addr if args.redis_addr else app_host_ip
    redis_port = args.redis_port
    redis_passwd = args.redis_passwd

    consul_addr = args.consul_addr if args.consul_addr else app_host_ip
    consul_port = args.consul_port

    logger.info('start ai server on %s:%d', host, args.port)
    logger.info('start consul server on %s:%d', consul_addr, consul_port)
    logger.info('start redis server on %s:%d: %s', redis_addr, redis_port, redis_passwd)

    thread = Thread(target=_delay_do_consul, args=(host, args.port))
    thread.start()

    try:
        g_redis = redis.StrictRedis(redis_addr, port=redis_port, password=redis_passwd)
    except Exception as err:
        logger.error('redis not connect: {}'.format(err))

    try:
        app.run(host=host, port=args.port)
    finally:
        app_quit = True
        thread.join()
