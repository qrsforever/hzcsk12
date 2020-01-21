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
import redis

from flask import Flask, request
from flask_cors import CORS
from threading import Thread

from k12ai_consul import k12ai_consul_init, k12ai_consul_register, k12ai_consul_service
from k12ai_platform import k12ai_platform_stats, k12ai_platform_control
from k12ai_errmsg import k12ai_error_message as _err_msg

if os.environ.get("K12AI_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)

g_app_quit = False
g_redis = None


def _delay_do_consul(host, port):
    time.sleep(3)
    while not g_app_quit:
        try:
            k12ai_consul_register('k12ai', host, port)
            break
        except Exception as err:
            logger.error("consul agent service register err", err)
            time.sleep(3)


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
        service_uuid = reqjson['service_uuid']
        service_params = reqjson.get('service_params', None)
        isasync = reqjson.get('async', False)
    except Exception:
        return json.dumps(_err_msg(100101, exc=True))

    try:
        code, msg = k12ai_platform_stats(op, user, service_uuid, service_params, isasync)
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
        service_uuid = reqjson['service_uuid']
        service_params = reqjson.get('service_params', None)
        isasync = reqjson.get('async', False)
    except Exception:
        return json.dumps(_err_msg(100101, exc=True))

    try:
        code, msg = k12ai_platform_control(op, user, service_uuid, service_params, isasync)
        return json.dumps(_err_msg(100202 if code < 0 else 100000, msg))
    except Exception:
        return json.dumps(_err_msg(100202, exc=True))


### Framework
@app.route('/k12ai/framework/schema', methods=['POST'])
def _framework_schema():
    logger.info('call _framework_schema')
    try:
        reqjson = json.loads(request.get_data().decode())
        service_name = reqjson['service_name']
        service_task = reqjson['service_task']
        dataset_name = reqjson['dataset_name']
        network_type = reqjson.get('network_type', 'base_model')
    except json.decoder.JSONDecodeError:
        return json.dumps(_err_msg(100103, request.get_data().decode()))
    except Exception:
        return json.dumps(_err_msg(100101, reqjson, exc=True))

    agent = k12ai_consul_service(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))
    try:
        code, msg = agent.schema(service_task, network_type, dataset_name)
        return json.dumps(_err_msg(code, msg))
    except Exception:
        return json.dumps(_err_msg(100207, exc=True))


@app.route('/k12ai/framework/execute', methods=['POST'])
def _framework_execute():
    logger.info('call _framework_execute')
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
        if isinstance(service_params, str):
            service_params = json.loads(service_params)
    except Exception:
        return json.dumps(_err_msg(100101, exc=True))

    if not isinstance(user, str) or not isinstance(service_uuid, str):
        return json.dumps(_err_msg(100102, 'user or service_uuid type not str'))

    agent = k12ai_consul_service(service_name)
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
        return "-1"
    return "0"


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
            default='127.0.0.1',
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

    k12ai_consul_init(args.consul_addr, args.consul_port, LEVEL == logging.DEBUG)

    thread = Thread(target=_delay_do_consul, args=(args.host, args.port))
    thread.start()

    try:
        g_redis = redis.StrictRedis(args.redis_addr,
                port=args.redis_port,
                password=args.redis_passwd)
    except Exception as err:
        logger.error('redis not connect: {}'.format(err))

    try:
        app.run(host=args.host, port=args.port)
    finally:
        g_app_quit = True
        thread.join()
