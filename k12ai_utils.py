#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_utils.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-07 09:45:23

import socket
import os

def k12ai_get_hostname():
    val = os.environ.get('HOST_NAME', None)
    if not val:
        return socket.gethostname()
    return val

def k12ai_get_hostip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8',80))
        return s.getsockname()[0]
    finally:
        s.close()
    return ''

def k12ai_get_netip():
    result = os.popen('curl -s http://txt.go.sohu.com/ip/soip| grep -P -o -i "(\d+\.\d+.\d+.\d+)"', 'r') # noqa
    if result:
        return result.read().strip('\n')
    return ''
