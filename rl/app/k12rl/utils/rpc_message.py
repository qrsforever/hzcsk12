#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file rpc_message.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-20 16:43

import os
import zerorpc

_RPCClient = None
_RPCEnable = -1
K12RL_TOKEN, K12RL_OP, K12RL_USER, K12RL_UUID = None, None, None, None


def hzcsk12_send_message(msgtype, message, end=False):
    global _RPCClient, _RPCEnable, K12RL_TOKEN, K12RL_OP, K12RL_USER, K12RL_UUID

    if _RPCEnable == 0:
        return

    if _RPCEnable == -1:
        host = os.environ.get('K12RL_RPC_HOST', None)
        port = os.environ.get('K12RL_RPC_PORT', None)
        if not host or not port:
            _RPCEnable = 0
            return
        K12RL_TOKEN = os.environ.get('K12RL_TOKEN', 'Unkown')
        K12RL_OP = os.environ.get('K12RL_OP', 'Unkown')
        K12RL_USER = os.environ.get('K12RL_USER', 'Unkown')
        K12RL_UUID = os.environ.get('K12RL_UUID', 'Unkown')
        _RPCClient = zerorpc.Client(
                connect_to='tcp://{}:{}'.format(host, port),  # noqa
                timeout=2,
                passive_heartbeat=True)
        _RPCEnable = 1

    try:
        if message:
            _RPCClient.send_message(K12RL_TOKEN, K12RL_OP, K12RL_USER, K12RL_UUID, msgtype, message)
        if end:
            _RPCClient.close()
    except Exception:
        pass
