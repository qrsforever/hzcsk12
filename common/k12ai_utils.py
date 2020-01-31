#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_utils.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-07 09:45:23

import socket
import os
import json


def k12ai_utils_topdir():
    return os.path.abspath(
                os.path.dirname(os.path.abspath(__file__)) + '/..')


def k12ai_utils_hostname():
    val = os.environ.get('HOST_NAME', None)
    if not val:
        return socket.gethostname()
    return val


def k12ai_utils_hostip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8',80))
        return s.getsockname()[0]
    finally:
        s.close()
    return ''


def k12ai_utils_netip():
    result = os.popen('curl -s http://txt.go.sohu.com/ip/soip| grep -P -o -i "(\d+\.\d+.\d+.\d+)"', 'r') # noqa
    if result:
        return result.read().strip('\n')
    return ''


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