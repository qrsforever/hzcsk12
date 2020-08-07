#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-15 18:53:37

import os, time, json
import argparse
import redis

from flask import Flask, request
from flask_cors import CORS
from threading import Thread

from k12ai.k12ai_utils import k12ai_timeit
from k12ai.k12ai_errmsg import k12ai_error_message as _err_msg
from k12ai.k12ai_consul import (k12ai_consul_init, k12ai_consul_register, k12ai_consul_service)
from k12ai.k12ai_platform import (k12ai_platform_stats, k12ai_platform_control)
from k12ai.k12ai_logger import (k12ai_set_loglevel, k12ai_set_logfile, Logger)

_DEBUG_ = True if os.environ.get("K12AI_DEBUG") else False
if _DEBUG_:
    k12ai_set_loglevel('debug')
k12ai_set_logfile('k12ai.log')


app = Flask(__name__)
CORS(app, supports_credentials=True)

g_app_quit = False
g_redis = None


def _delay_do_loop(host, port):
    time.sleep(3)
    while not g_app_quit:
        try:
            k12ai_consul_register('k12ai', host, port)
            break
        except Exception as err:
            Logger.error("consul agent service register err: {}".format(err))
            time.sleep(3)

    while not g_app_quit:
        try:
            k12ai_platform_stats('k12ai', 'query', 'admin', 'admin',
                {'cpus': True, 'gpus': True}, True)
        except Exception:
            pass
        time.sleep(30)


### Platform
@app.route('/k12ai/platform/stats', methods=['POST'], endpoint='platform_stats')
@k12ai_timeit(handler=Logger.debug)
def _platform_stats():
    try:
        reqjson = json.loads(request.get_data().decode())
        appId = reqjson.get('appId', 'k12ai')
        user = reqjson['user']
        op = reqjson['op']
        if op not in ('query'):
            return json.dumps(_err_msg(100102, f'not support op:{op}'))
        service_uuid = reqjson['service_uuid']
        service_params = reqjson.get('service_params', None)
        if isinstance(service_params, str):
            service_params = json.loads(service_params)
        isasync = reqjson.get('isasync', False)
    except Exception:
        return json.dumps(_err_msg(100101, exc=True))

    try:
        code, msg = k12ai_platform_stats(appId, op, user, service_uuid, service_params, isasync)
        return json.dumps(_err_msg(code, msg))
    except Exception:
        return json.dumps(_err_msg(100202, exc=True))


@app.route('/k12ai/platform/control', methods=['POST'], endpoint='platform_control')
@k12ai_timeit(handler=Logger.debug)
def _platform_control():
    try:
        reqjson = json.loads(request.get_data().decode())
        user = reqjson['user']
        op = reqjson['op']
        if op not in ('container.stop'):
            return json.dumps(_err_msg(100102, f'not support op:{op}'))
        service_uuid = reqjson.get('service_uuid', None)
        service_params = reqjson.get('service_params', None)
        if isinstance(service_params, str):
            service_params = json.loads(service_params)
        isasync = reqjson.get('async', False)
    except Exception:
        return json.dumps(_err_msg(100101, exc=True))

    try:
        code, msg = k12ai_platform_control(op, user, service_uuid, service_params, isasync)
        return json.dumps(_err_msg(code, msg))
    except Exception:
        return json.dumps(_err_msg(100202, exc=True))


### Framework
@app.route('/k12ai/framework/schema', methods=['POST'], endpoint='framework_schema')
@k12ai_timeit(handler=Logger.debug)
def _framework_schema():
    try:
        reqjson = json.loads(request.get_data().decode())
        version = reqjson.get('version', '0')
        levelid = reqjson.get('levelid', 0)
        service_name = reqjson['service_name']
        service_task = reqjson['service_task']
        dataset_name = reqjson['dataset_name']
        dataset_info = reqjson.get('dataset_info', "{}")
        network_type = reqjson['network_type']
        assert network_type != '', 'network_type'
    except json.decoder.JSONDecodeError:
        return json.dumps(_err_msg(100103, request.get_data().decode()))
    except AssertionError as verr:
        return json.dumps(_err_msg(100102, str(verr)))
    except KeyError as kerr:
        return json.dumps(_err_msg(100101, str(kerr)))
    except Exception as uerr:
        return json.dumps(_err_msg(100199, str(uerr)))

    agent = k12ai_consul_service(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))
    try:
        code, msg = agent.schema(version, levelid, service_task, network_type,
                dataset_name, json.loads(dataset_info))
        return json.dumps(_err_msg(code, msg))
    except Exception as err:
        print(f"{err}")
        return json.dumps(_err_msg(100207, exc=True))


@app.route('/k12ai/framework/memstat', methods=['POST'], endpoint='framework_memstat')
@k12ai_timeit(handler=Logger.debug)
def _framework_memstat():
    try:
        reqjson = json.loads(request.get_data().decode())
        service_name = reqjson['service_name']
        service_params = reqjson.get('service_params', None)
        if isinstance(service_params, str):
            service_params = json.loads(service_params)
    except json.decoder.JSONDecodeError:
        return json.dumps(_err_msg(100103, request.get_data().decode()))
    except AssertionError as verr:
        return json.dumps(_err_msg(100102, str(verr)))
    except KeyError as kerr:
        return json.dumps(_err_msg(100101, str(kerr)))
    except Exception as uerr:
        return json.dumps(_err_msg(100199, str(uerr)))

    agent = k12ai_consul_service(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))
    try:
        code, msg = agent.memstat(service_params)
        return json.dumps(_err_msg(code, msg))
    except Exception:
        return json.dumps(_err_msg(100207, exc=True))


@app.route('/k12ai/framework/execute', methods=['POST'], endpoint='framework_execcute')
@k12ai_timeit(handler=Logger.debug)
def _framework_execute():
    try:
        reqjson = json.loads(request.get_data().decode())
        appId = reqjson.get('appId', 'talentai')
        token = reqjson['token']
        user = reqjson['user']
        op = reqjson['op']
        if op not in ('train.start', 'train.stop',
                'train.pause', 'train.resume', 'train.rob',
                'evaluate.start', 'evaluate.stop',
                'predict.start', 'predict.stop',
                'runcode.start', 'runcode.stop'):
            return json.dumps(_err_msg(100102, f'not support op:{op}'))

        service_name = reqjson['service_name']
        service_uuid = reqjson['service_uuid']
        service_params = reqjson.get('service_params', None)
        if isinstance(service_params, str):
            service_params = json.loads(service_params)
        # TODO not good
        if service_name == 'k12cv':
            custom_model = reqjson.get('custom_model', None)
            if custom_model is not None:
                service_params['network.net_def'] = custom_model
    except Exception:
        return json.dumps(_err_msg(100101, exc=True))

    if not isinstance(user, str) or not isinstance(service_uuid, str):
        return json.dumps(_err_msg(100102, 'user or service_uuid type not str'))

    agent = k12ai_consul_service(service_name)
    if not agent:
        return json.dumps(_err_msg(100201, f'service name:{service_name}'))
    try:
        code, msg = agent.execute(appId, token, op, user, service_uuid, service_params)
        return json.dumps(_err_msg(code, msg))
    except Exception:
        return json.dumps(_err_msg(100202, exc=True))


@app.route('/k12ai/private/pushmsg', methods=['POST', 'GET'])
def _framework_message_push():
    # Logger.debug('call _framework_message')
    try:
        if g_redis:
            key = request.args.get("key", default='unknown')
            g_redis.lpush(key, request.get_data().decode())
    except Exception as err:
        Logger.info(err)
        return "-1"
    return "0"


@app.route('/k12ai/private/popmsg', methods=['GET'])
def _framework_message_pop():
    response = []
    try:
        if g_redis:
            key = request.args.get("key", default='unknown')
            while True:
                item = g_redis.rpop(key)
                if item is None:
                    break
                response.append(item.decode())
            return json.dumps(response)
    except Exception as err:
        Logger.info(err)
    finally:
        return json.dumps(response)


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

    k12ai_consul_init(args.consul_addr, args.consul_port, _DEBUG_)

    thread = Thread(target=_delay_do_loop, args=(args.host, args.port))
    thread.start()

    try:
        g_redis = redis.StrictRedis(args.redis_addr,
                port=args.redis_port,
                password=args.redis_passwd)
    except Exception as err:
        Logger.error('redis not connect: {}'.format(err))

    try:
        app.run(host=args.host, port=args.port)
    finally:
        g_app_quit = True
        thread.join()
