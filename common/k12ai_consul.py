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
g_consul_debug = False


class RPCServiceAgent(object):
    def __init__(self, addr, port, timeout):
        self._addr = addr
        self._port = port
        self._timeout = timeout

    def __getattr__(self, method):
        def __wrapper(*args, **kwargs):
            try:
                client = zerorpc.Client(timeout=self._timeout)
                client.connect('tcp://{}:{}'.format(self._addr, self._port))
                result = client(method, *args, **kwargs)
                client.close()
                return result
            except Exception as err:
                raise err
        return __wrapper


def k12ai_consul_init(addr, port, debug=False):
    global g_consul_addr, g_consul_port, g_consul_debug 
    g_consul_addr = addr
    g_consul_port = port
    g_consul_debug = debug


def k12ai_consul_register(name, host, port, timeout=5):
    print(g_consul_addr, g_consul_port)
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


def k12ai_consul_message(op, user, uuid, msgtype, message, clear=False):
    client = consul.Consul(g_consul_addr, port=g_consul_port)
    service = client.agent.services().get('k12ai')
    if not service:
        print("Not found k12ai service!")
        return

    data = {
            'version': '0.1.0',
            'type': msgtype,
            'tag': 'k12ai',
            'user': user,
            'service_uuid': uuid,
            'op': op
            } # noqa

    now_time = time.time()
    data['timestamp'] = round(now_time * 1000)
    data['datetime'] = time.strftime('%Y%m%d%H%M%S', time.localtime(now_time))
    data['data'] = message

    # service
    api = 'http://{}:{}/k12ai/private/message'.format(service['Address'], service['Port'])
    requests.post(api, json=data)

    if g_consul_debug:
        if clear:
            client.kv.delete('framework/%s/%s' % (user, uuid), recurse=True)
        key = 'framework/%s/%s/%s/%s/%s' % (user, uuid, op, msgtype, data['datetime'])
        client.kv.put(key, json.dumps(data, indent=2))
