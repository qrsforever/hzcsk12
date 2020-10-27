#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_consul.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-21 14:37

import json
import time
import zerorpc
import consul
import requests

g_consul_addr = "127.0.0.1"
g_consul_port = 8500
g_consul_debug = True
g_errors_store = True


class RPCServiceAgent(object):
    def __init__(self, addr, port, timeout):
        self._addr = 'localhost' if not addr else addr
        self._port = port
        self._timeout = timeout

    def __getattr__(self, method):
        def __wrapper(*args, **kwargs):
            try:
                client = zerorpc.Client(timeout=self._timeout)
                client.connect('tcp://{}:{}'.format(self._addr, self._port))
                result = client(method, *args, **kwargs)
                return result
            except Exception as err:
                raise err
            finally: # fixbug: ZMQError Too many open files
                if client:
                    client.close()
        return __wrapper


def k12ai_consul_init(addr, port, debug=False):
    global g_consul_addr, g_consul_port, g_consul_debug
    g_consul_addr = addr
    g_consul_port = port
    g_consul_debug = debug


def k12ai_consul_register(name, host, port, timeout=5):
    # TODO using configuration
    return
    client = consul.Consul(host=g_consul_addr, port=g_consul_port)
    client.agent.service.register(
            name=name, address=host, port=port, tags=('k12ai',),
            check=consul.Check.tcp(host, port,
                interval='%ds'%(2 * timeout),
                timeout='%ds'%timeout,
                deregister='%ds' % (2 * timeout)))


def k12ai_consul_service(name, timeout=15):
    client = consul.Consul(g_consul_addr, port=g_consul_port)
    service = client.agent.services().get(name)
    if not service:
        return None
    else:
        return RPCServiceAgent(service['Address'], service['Port'], timeout)


def k12ai_consul_message(sname, appId, token, op, user, uuid, msgtype, message, clear=False):
    client = consul.Consul(g_consul_addr, port=g_consul_port)
    service = client.agent.services().get('k12ai')
    if not service:
        print("Not found k12ai service!")
        return

    server = f'{service["Address"]}:{service["Port"]}'

    data = {
        'version': '0.1.0',
        'server': server,
        'type': msgtype,
        'appId': appId,
        'token': token,
        'user': user,
        'op': op,
        'service_name': sname,
        'service_uuid': uuid,
    }

    now_time = time.time()
    data['timestamp'] = round(now_time * 1000)
    data['datetime'] = time.strftime('%Y%m%d%H%M%S', time.localtime(now_time))
    data['data'] = message

    # service
    api = 'http://{}/k12ai/private/pushmsg?key={}.{}'.format(server, appId, msgtype)
    requests.post(api, json=data)

    if g_consul_debug:
        if clear:
            client.kv.delete('framework/%s/%s' % (user, uuid), recurse=True)
        key = 'framework/%s/%s/%s/%s/%s' % (user, uuid, op.split('.')[0], msgtype, data['datetime'])
        jsondata = json.dumps(data, indent=2)
        if len(jsondata) < 512000:
            client.kv.put(key, jsondata)
    if g_errors_store and msgtype == 'error' and message['code'] > 100100:
        client.kv.put('errors/%s/%s/%s' % (user, uuid, data['datatime']), json.dumps(data, indent=2))
