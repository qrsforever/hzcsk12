#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file rpc_message.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-12 14:56

import os
import sys
import zerorpc
import traceback

_RPCClient = None
_RPCEnable = -1
K12ML_TOKEN, K12ML_OP, K12ML_USER, K12ML_UUID = None, None, None, None


def _rpc_send_message(msgtype, message, end=False):
    global _RPCClient, _RPCEnable, K12ML_TOKEN, K12ML_OP, K12ML_USER, K12ML_UUID

    if _RPCEnable == 0:
        return

    if _RPCEnable == -1:
        host = os.environ.get('K12ML_RPC_HOST', None)
        port = os.environ.get('K12ML_RPC_PORT', None)
        if not host or not port:
            _RPCEnable = 0
            return
        K12ML_TOKEN = os.environ.get('K12ML_TOKEN', 'Unkown')
        K12ML_OP = os.environ.get('K12ML_OP', 'Unkown')
        K12ML_USER = os.environ.get('K12ML_USER', 'Unkown')
        K12ML_UUID = os.environ.get('K12ML_UUID', 'Unkown')
        _RPCClient = zerorpc.Client(
                connect_to='tcp://{}:{}'.format(host, port),  # noqa
                timeout=2,
                passive_heartbeat=True)
        _RPCEnable = 1

    try:
        if message:
            _RPCClient.send_message(K12ML_TOKEN, K12ML_OP, K12ML_USER, K12ML_UUID, msgtype, message)
        if end:
            _RPCClient.close()
    except Exception:
        pass


def hzcsk12_send_message(errmsg):
    if isinstance(errmsg, dict):
        _rpc_send_message('metrics', errmsg)
        return
    if errmsg.startswith('k12ml_running'):
        _rpc_send_message('status', {'value': 'running'})
        print(errmsg)
        return

    if errmsg.startswith('k12ml_finish'):
        _rpc_send_message('status', {'value': 'exit', 'way': 'finish'})
        print(errmsg)
        return

    if errmsg.startswith('k12ml_error'):
        _rpc_send_message('error', errmsg)
        _rpc_send_message('status', {'value': 'exit', 'way': 'error'})
        print(errmsg)
        return

    if errmsg.startswith('k12ml_except'):
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        exc_type, exc_value, exc_tb = sys.exc_info()
        message = { # noqa: E126
                'filename': filename,
                'linenum': lineno,
                'err_type': exc_type.__name__,
                'err_text': str(exc_value)
                }
        message['trackback'] = []
        tbs = traceback.extract_tb(exc_tb)
        for tb in tbs:
            err = { # noqa: E126
                    'filename': tb.filename,
                    'linenum': tb.lineno,
                    'funcname': tb.name,
                    'source': tb.line
                    }
            message['trackback'].append(err)
        _rpc_send_message('error', message)
        _rpc_send_message('status', {'value': 'exit', 'way': 'crash'})
        print(message)
