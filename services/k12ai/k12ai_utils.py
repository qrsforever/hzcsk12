#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_utils.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-07 09:45:23

import time
import socket
import os
import json

_LANIP = None
_NETIP = None


def k12ai_timeit(handler):
    def decorator(func):
        def timed(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            if handler:
                handler('"{}" took {:.3f} ms to execute'.format(func.__name__, (te - ts) * 1000))
            return result 
        return timed
    return decorator


def k12ai_utils_topdir():
    return os.path.abspath(
                os.path.dirname(os.path.abspath(__file__)) + '/../..')


def k12ai_utils_hostname():
    val = os.environ.get('HOST_NAME', None)
    if val:
        return val
    return socket.gethostname()


def k12ai_utils_lanip():
    global _LANIP
    if _LANIP:
        return _LANIP
    val = os.environ.get('HOST_LANIP', None)
    if not val:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8',80))
            val = s.getsockname()[0]
        finally:
            s.close()
    _LANIP = val
    return _LANIP 


def k12ai_utils_netip():
    global _NETIP
    if _NETIP:
        return _NETIP
    val = os.environ.get('HOST_NETIP', None)
    if not val:
        result = os.popen('curl -s http://txt.go.sohu.com/ip/soip| grep -P -o -i "(\d+\.\d+.\d+.\d+)"', 'r') # noqa
        if result:
            val = result.read().strip('\n')
    _NETIP = val
    return _NETIP


def k12ai_utils_diff(conf1, conf2):
    if isinstance(conf1, dict):
        param1 = conf1
    else:
        if not os.path.exists(conf1):
            return True
        with open(conf1, 'r') as f1:
            param1 = json.loads(f1.read())

    if isinstance(conf2, dict):
        param2 = conf2
    else:
        if not os.path.exists(conf2):
            return True
        with open(conf2, 'r') as f2:
            param2 = json.loads(f2.read())

    diff = False
    for key in param1.keys() - param2.keys():
        print(f"Key '{key}' found in training configuration but not in the serialization "
                     f"directory we're recovering from.")
        diff = True

    for key in param1.keys() - param2.keys():
        print(f"Key '{key}' found in the serialization directory we're recovering from "
                     f"but not in the training config.")
        diff = True

    for key in param1.keys():
        if param1.get(key, None) != param2.get(key, None):
            print(f"Value for '{key}' in training configuration does not match that the value in "
                         f"the serialization directory we're recovering from: "
                         f"{param1[key]} != {param2[key]}")
            diff = True
    return diff
