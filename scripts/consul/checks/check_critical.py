#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import json
# import hashlib
# from urllib.parse import quote_plus, unquote_plus
from urllib.request import urlopen

def debug_out(fname, data):
    with open('/tmp/%s.txt' % fname, 'a') as fw:
        fw.write(data)

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

    content = {}
    for item in data:
        node = item["Node"]
        name = item["ServiceName"]
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
