#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import socket
import hashlib
from urllib.parse import quote_plus
from urllib.request import urlopen

DEBUG = True

PHONES = [
    '15801310416'
]

MSG_URI = 'http://115.231.168.138:8861'
UID = '966646'
CODE = '4YC3HDAZRO'
SRCPHONE = '88191008369646'

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
        if tcp_port_check(node, int(item['CheckID'][-4:])):
            continue
        else:
            if node not in content:
                content[node] = []
            content[node].append(name)
    if len(content) == 0:
        return
    debug_out('report_content', json.dumps(content))

    report_context = '【TalentAI远程监控】{} <-验证码-> '.format(json.dumps(content))

    report_data = []
    for tel in PHONES:
        report_data.append({
            "phone": tel,
            "context": report_context})
    msg = quote_plus(json.dumps(report_data, ensure_ascii=False, separators=(',', ':')))
    sign = hashlib.md5('{}{}'.format(msg, CODE).encode()).hexdigest()
    request = '{}?uid={}&msg={}&sign={}&srcphone={}'.format(MSG_URI, UID, msg, sign, SRCPHONE)
    debug_out('report_content', request)

    urlopen(request)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        debug_out('err', '{}'.format(err))

    print('0')
