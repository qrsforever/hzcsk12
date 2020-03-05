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
import GPUtil
import psutil

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


def k12ai_memstat_message():
    # not all gpus
    message = {
        'app_cpu_memory_usage_MB': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
        'app_gpu_memory_usage_MB': round(torch.cuda.max_memory_allocated() / 1024**2, 3),
        'sys_cpu_memory_free_MB': round(psutil.virtual_memory().available / 1024**2, 3),
        'sys_gpu_memory_free_MB': round(GPUtil.getGPUs()[0].memoryFree, 3),
        'app_cpu_max_memory_children_MB': round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 3),
        'app_gpu_max_memory_cached_MB': round(torch.cuda.max_memory_cached() / 1024**2, 3),
        'app_gpu_memory_allocated_MB': round(torch.cuda.memory_allocated() / 1024**2, 3),
        'app_gpu_memory_cached_MB': round(torch.cuda.memory_cached() / 1024**2, 3)
    }
    return message


class MessageReport(object):
    RUNNING = 1
    ERROR = 2
    EXCEPT = 3
    FINISH = 4

    @staticmethod
    def status(what, msg=None):
        if what == MessageReport.RUNNING:
            k12ai_send_message('status', {'value': 'running'})
            return

        if what == MessageReport.ERROR:
            k12ai_send_message('error', msg or {})
            k12ai_send_message('status', {'value': 'exit', 'way': 'error'})
            return

        if what == MessageReport.EXCEPT:
            k12ai_send_message('error', msg or k12ai_except_message())
            k12ai_send_message('status', {'value': 'exit', 'way': 'crash'})
            return

        if what == MessageReport.FINISH:
            k12ai_send_message('status', {
                'value': 'exit',
                'way': 'finish',
                'memstat': k12ai_memstat_message()
            })
            return

    @staticmethod
    def metrics(metrics, memstat=False, end=False):
        if memstat:
            metrics['memstat'] = k12ai_memstat_message()
        k12ai_send_message('metrics', metrics, end)
