#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import socket
# import hashlib
# from urllib.parse import quote_plus, unquote_plus
from urllib.request import urlopen

DEBUG = True

def debug_out(fname, data):
    if DEBUG:
        with open('/tmp/%s.txt' % fname, 'a') as fw:
            fw.write(data)
            fw.write('\n')

def tcp_port_check(ip, port, timeout=2.0):
    try:
        s = socket.socket()
        s.settimeout(timeout)
        s.connect((ip, port))
        s.close()
        return True
    except socket.error as err:
        return False

def main():
    host = os.environ.get('CONSUL_ADDR', None)
    port = os.environ.get('CONSUL_PORT', None)
    if host is None or port is None:
        return

    response = urlopen('http://{}:{}/v1/status/leader'.format(host, port))
    if host not in response.read().decode('utf-8'):
        return

    data = json.load(sys.stdin)
    if len(data) == 0:
        return
    debug_out('trigger_data', json.dumps(data))

    content = {}
    for item in data:
        node = item["Node"]
        name = item["ServiceName"]
        if tcp_port_check(node, int(item['ServiceID'])):
            debug_out('tcp_ok', json.dumps(item))
            continue
        else:
            if node not in content:
                content[node] = []
            content[node].append(name)
    debug_out('report_content', json.dumps(content))

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        debug_out('err', '{}'.format(err))

    print('0')
