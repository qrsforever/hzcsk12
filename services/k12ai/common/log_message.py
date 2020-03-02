#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_message.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-01 23:56

import sys
import traceback
import resource
import torch

from k12ai.common.rpc_message import k12ai_send_message


def k12ai_except_message():
    exc_type, exc_value, exc_tb = sys.exc_info()
    message = {
        'err_type': exc_type.__name__,
        'err_text': str(exc_value)
    }
    message['trackback'] = []
    tbs = traceback.extract_tb(exc_tb)
    for tb in tbs:
        err = {
            'filename': tb.filename,
            'linenum': tb.lineno,
            'funcname': tb.name,
            'source': tb.line
        }
        message['trackback'].append(err)
    return message


def k12ai_memory_message(device=None):
    message = {}
    message['cpu_max_memory_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    message['gpu_max_memory_allocated'] = torch.cuda.max_memory_allocated(device)
    message['gpu_max_memory_cached'] = torch.cuda.max_memory_cached(device)
    message['gpu_memory_allocated'] = torch.cuda.memory_allocated(device)
    message['gpu_memory_cached'] = torch.cuda.memory_cached(device)
    return message


def k12ai_status_message(what, msg=None):
    if what.startswith('k12ai_metrics'):
        k12ai_send_message('metrics', msg)
        return

    if what.startswith('k12ai_running'):
        k12ai_send_message('status', {'value': 'running'})
        return

    if what.startswith('k12ai_finish'):
        k12ai_send_message('status', {
            'value': 'exit',
            'way': 'finish',
            'memory': k12ai_memory_message()})
        return

    if what.startswith('k12ai_error'):
        k12ai_send_message('error', msg)
        k12ai_send_message('status', {'value': 'exit', 'way': 'error'})
        print(msg)
        return

    if what.startswith('k12ai_except'):
        message = k12ai_except_message()
        k12ai_send_message('error', message)
        k12ai_send_message('status', {'value': 'exit', 'way': 'crash'})
        print(message)
