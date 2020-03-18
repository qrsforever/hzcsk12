#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_message.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-01 23:56

import os
import sys
import time
import traceback
import GPUtil
import psutil
import numpy
import torch

from torch.cuda import (max_memory_allocated, memory_allocated, max_memory_cached, memory_cached)
from resource import (getrusage, RUSAGE_SELF, RUSAGE_CHILDREN)
from k12ai.common.rpc_message import k12ai_send_message
from k12ai.common.util_misc import base64_image

import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

g_starttime = None
g_memstat = {}

g_runbynb = True if os.environ.get('K12AI_RUN_BYNB', None) else False
g_tbwriter = None


def _get_writer():
    global g_tbwriter
    if g_runbynb:
        if not g_tbwriter:
            g_tbwriter = SummaryWriter(log_dir='/cache/tblogs')
    return g_tbwriter


def _tensor_to_list(x):
    return x.cpu().numpy().astype(float).reshape(-1).tolist()


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
    sys_gpu_mfree = []
    for i, g in enumerate(GPUtil.getGPUs(), 0):
        _peak_update(f'peak_gpu_{i}_memory_cached_MB', memory_cached(i), 2)
        _peak_update(f'peak_gpu_{i}_memory_allocated_MB', memory_allocated(i), 2)
        _peak_update(f'peak_gpu_{i}_max_memory_cached_MB', max_memory_cached(i), 2)
        app_gpu_usage += _peak_update(f'peak_gpu_{i}_max_memory_allocated_MB', max_memory_allocated(i), 2)
        sys_gpu_mfree.append(round(GPUtil.getGPUs()[i].memoryFree, 3))

    return {
        'app_cpu_memory_usage_MB': app_cpu_usage,
        'app_gpu_memory_usage_MB': app_gpu_usage,
        'sys_cpu_memory_free_MB': round(psutil.virtual_memory().available / 1024**2, 3),
        'sys_gpu_memory_free_MB': sys_gpu_mfree,
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


class MessageMetric(object):
    def __init__(self):
        self._metrics = []
        self._writer = _get_writer()

    @property
    def data(self):
        return self._metrics

    def send(self):
        if len(self._metrics) > 0:
            k12ai_send_message('metrics', self._metrics)
            self._metrics = []

    def _mmjson(self, ty, tag, title, value, width, height):
        obj = {
            'type': ty,
            'name': tag,
            'data': {
                'title': title,
                'value': value
            }
        }
        if width:
            obj['width'] = width
        if height:
            obj['height'] = height
        return obj

    def add_scalar(self, tag, x, y, width=None, height=None):
        if isinstance(x, dict) and isinstance(y, dict):
            obj = {
                'type': 'scalar',
                'name': tag,
                'data': {
                    'x': x,
                    'y': y
                }
            }
            if width:
                obj['width'] = width
            if height:
                obj['height'] = height
            self._metrics.append(obj)
        if self._writer:
            self._writer.add_scalar(tag, y, x)
        return self

    def add_scalars(self, tag, x, y, width=None, height=None):
        if self._writer:
            self._writer.add_scalars(tag, y, x)
        if isinstance(x, dict) and isinstance(y, dict):
            obj = {
                'type': 'scalars',
                'name': tag,
                'data': {
                    'x': x,
                    'y': y
                }
            }
            if width:
                obj['width'] = width
            if height:
                obj['height'] = height
            self._metrics.append(obj)
        return self

    def add_image(self, tag, title, image, fmt='base64', step=None, width=None, height=None):
        if isinstance(image, str):
            value = image
        elif isinstance(image, (torch.Tensor, numpy.array)):
            if fmt == 'base64':
                value = base64_image(image)
            else:
                raise NotImplementedError
            if self._writer:
                self._writer.add_image(f'{tag}/{title}', image, step)
        obj = self._mmjson('image', tag, title, value, width, height)
        obj['format'] = fmt
        self._metrics.append(obj)
        return self

    def add_images(self, tag, data, fmt='base64', width=None, height=None):
        obj = {
            'type': 'images',
            'name': tag,
            'format': fmt,
            'data': data
        }
        if width:
            obj['width'] = width
        if height:
            obj['height'] = height
        self._metrics.append(obj)
        return self

    def add_matrix(self, tag, title, value, step=None, width=None, height=None):
        if self._writer:
            fig, ax = plt.subplots(figsize=(12,8))
            sns.heatmap(value, annot=True, fmt='d', linewidth=0.5,cmap='Blues', ax=ax, cbar=True)
            self._writer.add_figure(f'{tag}/{title}', fig, step, close=True)
        obj = self._mmjson('matrix', tag, title, value, width, height)
        self._metrics.append(obj)
        return self

    def add_text(self, tag, title, value, step=None, width=None, height=None):
        if self._writer:
            self._writer.add_text(f'{tag}/{title}', f'{value}', step)
        obj = self._mmjson('text', tag, title, value, width, height)
        self._metrics.append(obj)
        return self

    def add_histogram(self, tag, title, value, step=None, width=None, height=None):
        if self._writer:
            self._writer.add_histogram(f'{tag}/{title}', value, step)
        return self
