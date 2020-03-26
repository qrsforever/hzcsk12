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
import shutil # noqa
import numpy
import torch
import hashlib

from torch.cuda import (max_memory_allocated, memory_allocated, max_memory_cached, memory_cached)
from resource import (getrusage, RUSAGE_SELF, RUSAGE_CHILDREN)
from k12ai.common.rpc_message import k12ai_send_message
from k12ai.common.util_misc import base64_image

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter

g_starttime = None
g_memstat = {}

g_runbynb = True if os.environ.get('K12AI_RUN_BYNB', None) else False
g_tbwriter = None


def _get_writer():
    global g_tbwriter
    if g_runbynb:
        if not g_tbwriter:
            logdir = '/cache/tblogs'
            # if os.path.exists(logdir):
            #     shutil.rmtree(logdir, ignore_errors=True)
            #     os.mkdir(logdir)
            g_tbwriter = SummaryWriter(log_dir=logdir)
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
    STOP = 4
    FINISH = 5

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

        if what == MessageReport.EXCEPT:
            k12ai_send_message('error', {
                'status': 'crash',
                'errinfo': msg or k12ai_except_message()
            })
            return

        if what == MessageReport.STOP:
            k12ai_send_message('error', {
                'status': 'stop',
                'errinfo': msg or {'by comamnd way'}
            })
            sys.exit(0) # TODO
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
        return
        if memstat:
            metrics['memstat'] = k12ai_memstat_message()
        k12ai_send_message('metrics', metrics, end)


class MessageMetric(object):
    def __init__(self):
        self._metrics = []
        self._writer = _get_writer()
        self._cache_ids = {}

    @property
    def data(self):
        return self._metrics

    def send(self):
        if len(self._metrics) > 0:
            k12ai_send_message('metrics', self._metrics)
            self._metrics = []
        if self._writer:
            self._writer.flush()

    def _mmjson(self, ty, category, title, payload, width, height):
        def _get_id():
            key = f'{ty}{category}{title}'
            if key not in self._cache_ids.keys():
                self._cache_ids[key] = hashlib.md5(key.encode()).hexdigest()[0:16]
            return self._cache_ids[key]

        obj = {
            '_id_': _get_id(),
            'category': category,
            'type': ty,
            'data': {
                'title': title,
                'payload': payload,
            }
        }
        if width:
            obj['width'] = width
        if height:
            obj['height'] = height
        return obj

    def add_scalar(self, category, title, x, y, width=None, height=None):
        payload = {'x':{}, 'y':[]}
        if isinstance(x, dict):
            x = list(x.values())[0]
            payload['x']['label'] = list(x.keys())[0]
            payload['x']['value'] = x
        elif isinstance(x, int):
            payload['x']['label'] = 'iteration'
            payload['x']['value'] = x
        else:
            raise NotImplementedError
        if isinstance(y, dict):
            if len(y) == 0:
                return self
            for key, val in y.items():
                payload['y'].append({'label': key, 'value': val})
            if self._writer:
                self._writer.add_scalars(f'{category}/{title}', y, x)
        else:
            if isinstance(y, (int, float)):
                payload['y'].append({'label': title, 'value': y})
                if self._writer:
                    self._writer.add_scalar(f'{category}/{title}', y, x)
            elif isinstance(y, (list, tuple)) and len(y) == 2:
                if title in ('loss', 'acc'):
                    payload['y'].append({'label': f'train_{title}', 'value': y[0]})
                    payload['y'].append({'label': f'validation_{title}', 'value': y[1]})
                    if self._writer:
                        self._writer.add_scalars(f'{category}/{title}', {
                            f'train_{title}': y[0],
                            f'validation_{title}': y[1]}, x)
                else:
                    NotImplementedError
            else:
                NotImplementedError

        obj = self._mmjson('scalar', category, title, payload, width, height)
        self._metrics.append(obj)
        return self

    def add_image(self, category, title, image, fmt='base64string', step=None, width=None, height=None):
        if self._writer:
            if isinstance(image, Figure):
                self._writer.add_figure(f'{category}/{title}', image, step, close=True)
            elif isinstance(image, (torch.Tensor, numpy.ndarray)):
                self._writer.add_image(f'{category}/{title}', image, step)
            elif isinstance(image, bytes):
                from io import BytesIO
                from PIL import Image
                try:
                    image = Image.open(BytesIO(image))
                    self._writer.add_image(f'{category}/{title}', numpy.asarray(image), step, dataformats='HWC')
                except Exception as err:
                    print('{}'.format(err))
            else:
                raise NotImplementedError
        if fmt == 'base64string':
            payload = base64_image(image)
        elif fmt == 'path':
            payload = image
        else:
            raise NotImplementedError
        obj = self._mmjson('image', category, title, payload, width, height)
        obj['data']['format'] = fmt
        self._metrics.append(obj)
        return self

    def add_matrix(self, category, title, value, step=None, width=None, height=None):
        if self._writer:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(value, annot=True, fmt='d', linewidth=0.5,cmap='Blues', ax=ax, cbar=True)
            self._writer.add_figure(f'{category}/{title}', fig, step, close=True)
        if isinstance(value, numpy.ndarray):
            value = value.tolist()
        obj = self._mmjson('matrix', category, title, value, width, height)
        self._metrics.append(obj)
        return self

    def add_text(self, category, title, value, step=None, width=None, height=None):
        if self._writer:
            self._writer.add_text(f'{category}/{title}', f'{value}', step)
        obj = self._mmjson('text', category, title, value, width, height)
        self._metrics.append(obj)
        return self

    def add_histogram(self, category, title, value, step=None, width=None, height=None):
        if self._writer:
            self._writer.add_histogram(f'{category}/{title}', value, step)
        return self

    def add_graph(self, model, iimg=None):
        if self._writer:
            self._writer.add_graph(model, iimg)
        return self
