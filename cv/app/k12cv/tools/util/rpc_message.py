#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file rpc_message.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 18:48:57

import os
import zerorpc

_RPCClient = None
_RPCEnable = -1
K12CV_OP, K12CV_USER, K12CV_UUID = None, None, None

def hzcsk12_send_message(msgtype, message, end=False):
    global _RPCClient, _RPCEnable, K12CV_OP, K12CV_USER, K12CV_UUID

    if _RPCEnable == 0:
        return

    if _RPCEnable == -1:
        host = os.environ.get('K12CV_RPC_HOST', None)
        port = os.environ.get('K12CV_RPC_PORT', None)
        if not host or not port:
            _RPCEnable = 0
            return
        K12CV_OP = os.environ.get('K12CV_OP', 'Unkown')
        K12CV_USER = os.environ.get('K12CV_USER', 'Unkown')
        K12CV_UUID = os.environ.get('K12CV_UUID', 'Unkown')
        _RPCClient = zerorpc.Client(
                connect_to='tcp://{}:{}'.format(host, port),  # noqa
                timeout=2,
                passive_heartbeat=True)
        _RPCEnable = 1

    try:
        if message:
            _RPCClient.send_message(K12CV_OP, K12CV_USER, K12CV_UUID, msgtype, message)
        if end:
            _RPCClient.close()
    except Exception:
        pass
