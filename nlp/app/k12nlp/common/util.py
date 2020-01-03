#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file rpc_message.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 18:48:57

import os
import sys
import zerorpc
import traceback

_RPCClient = None
_RPCEnable = -1
K12NLP_OP, K12NLP_USER, K12NLP_UUID = None, None, None

def hzcsk12_send_message(msgtype, message, end=False):
    global _RPCClient, _RPCEnable, K12NLP_OP, K12NLP_USER, K12NLP_UUID

    if _RPCEnable == 0:
        return

    if _RPCEnable == -1:
        host = os.environ.get('K12NLP_RPC_HOST', None)
        port = os.environ.get('K12NLP_RPC_PORT', None)
        if not host or not port:
            _RPCEnable = 0
            return
        K12NLP_OP = os.environ.get('K12NLP_OP', 'Unkown')
        K12NLP_USER = os.environ.get('K12NLP_USER', 'Unkown')
        K12NLP_UUID = os.environ.get('K12NLP_UUID', 'Unkown')
        _RPCClient = zerorpc.Client(
                connect_to='tcp://{}:{}'.format(host, port),  # noqa
                timeout=2,
                passive_heartbeat=True)
        _RPCEnable = 1

    try:
        if message:
            _RPCClient.send_message(K12NLP_OP, K12NLP_USER, K12NLP_UUID, msgtype, message)
        if end:
            _RPCClient.close()
    except Exception:
        pass

def hzcsk12_error_message(errmsg=None, exc=False):
    if exc:
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
                    'souce': tb.line
                    }
            message['trackback'].append(err)
        hzcsk12_send_message('error', message)
        hzcsk12_send_message('status', {'value': 'exit', 'way': 'crash'})
    else:
        if errmsg:
            hzcsk12_send_message('error', errmsg)
            hzcsk12_send_message('status', {'value': 'exit', 'way': 'error'})
        else:
            print('UnkownError')
