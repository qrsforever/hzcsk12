#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file rpc_message.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 18:48:57

import os
import zerorpc

_DEBUG_ = True
_RPCClient = None
_RPCEnable = -1
K12AI_TOKEN, K12AI_OP, K12AI_USER, K12AI_UUID = None, None, None, None


def k12ai_send_message(msgtype, message, end=False):
    global _RPCClient, _RPCEnable, K12AI_TOKEN, K12AI_OP, K12AI_USER, K12AI_UUID

    if _RPCEnable == 0:
        return

    if _RPCEnable == -1:
        host = os.environ.get('K12AI_RPC_HOST', None)
        port = os.environ.get('K12AI_RPC_PORT', None)
        if not host or not port:
            _RPCEnable = 0
            return
        K12AI_TOKEN = os.environ.get('K12AI_TOKEN', 'Unkown')
        K12AI_OP = os.environ.get('K12AI_OP', 'Unkown')
        K12AI_USER = os.environ.get('K12AI_USER', 'Unkown')
        K12AI_UUID = os.environ.get('K12AI_UUID', 'Unkown')
        _RPCClient = zerorpc.Client(
                connect_to='tcp://{}:{}'.format(host, port),  # noqa
                timeout=2,
                passive_heartbeat=True)
        _RPCEnable = 1

    try:
        if message:
            _RPCClient.send_message(K12AI_TOKEN, K12AI_OP, K12AI_USER, K12AI_UUID, msgtype, message)
        if end:
            _RPCClient.close()

        if _DEBUG_ or msgtype == 'error':
            print(message)
    except Exception as err:
        if _DEBUG_:
            print(err)
