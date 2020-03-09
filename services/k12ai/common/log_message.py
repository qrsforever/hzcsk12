#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_message.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-01 23:56

import sys
import time
import traceback
import GPUtil
import psutil

from torch.cuda import (max_memory_allocated, memory_allocated, max_memory_cached, memory_cached)
from resource import (getrusage, RUSAGE_SELF, RUSAGE_CHILDREN)
from k12ai.common.rpc_message import k12ai_send_message

g_starttime = None
g_memstat = {}


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
    global g_memstat

    def _peak_update(key, value, flg):
        denom = 1024**2 if flg == 2 else 1024
        g_memstat[key] = max(g_memstat.get(key, 0), round(value / denom, 3))
        return g_memstat[key]

    app_cpu_usage = 0.0
    app_cpu_usage += _peak_update('peak_cpu_self_ru_maxrss', getrusage(RUSAGE_SELF).ru_maxrss, 1)
    app_cpu_usage += _peak_update('peak_cpu_children_ru_maxrss', getrusage(RUSAGE_CHILDREN).ru_maxrss, 1)
    app_gpu_usage = 0.0
    for i, g in enumerate(GPUtil.getGPUs(), 0):
        _peak_update(f'peak_gpu_{i}_memory_cached_MB', memory_cached(i), 2)
        _peak_update(f'peak_gpu_{i}_memory_allocated_MB', memory_allocated(i), 2)
        _peak_update(f'peak_gpu_{i}_max_memory_cached_MB', max_memory_cached(i), 2)
        app_gpu_usage += _peak_update(f'peak_gpu_{i}_max_memory_allocated_MB', max_memory_allocated(i), 2)

    return {
        'app_cpu_memory_usage_MB': app_cpu_usage,
        'app_gpu_memory_usage_MB': app_gpu_usage,
        'sys_cpu_memory_free_MB': round(psutil.virtual_memory().available / 1024**2, 3),
        'sys_gpu_memory_free_MB': round(GPUtil.getGPUs()[0].memoryFree, 3),
        **g_memstat
    }


class MessageReport(object):
    RUNNING = 1
    ERROR = 2
    EXCEPT = 3
    FINISH = 4

    @staticmethod
    def status(what, msg=None):
        if what == MessageReport.RUNNING:
            global g_starttime
            g_starttime = time.time()
            k12ai_send_message('error', {
                'status': 'running',
                'memstat': k12ai_memstat_message()
            })
            return

        if what == MessageReport.ERROR:
            k12ai_send_message('error', {
                'status': 'error',
                'errinfo': msg or {}
            })
            return

        if what == MessageReport.EXCEPT:
            k12ai_send_message('error', {
                'status': 'crash',
                'errinfo': msg or k12ai_except_message()
            })
            return

        if what == MessageReport.FINISH:
            k12ai_send_message('error', {
                'status': 'finish',
                'uptime': int(time.time() - g_starttime),
                'memstat': k12ai_memstat_message()
            })
            return

    @staticmethod
    def metrics(metrics, memstat=False, end=False):
        if memstat:
            metrics['memstat'] = k12ai_memstat_message()
        k12ai_send_message('metrics', metrics, end)
