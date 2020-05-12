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
import torch # noqa
import hashlib
import base64
import io

from PIL import Image

from torch.cuda import (max_memory_allocated, memory_allocated, max_memory_cached, memory_cached)
from resource import (getrusage, RUSAGE_SELF, RUSAGE_CHILDREN)
from k12ai.common.rpc_message import k12ai_send_message
from k12ai.common.util_misc import ( # noqa
    image2bytes,
    handle_exception,
    dr_scatter3D, 
    dr_scatter2D,
)

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure # noqa
from torch.utils.tensorboard import SummaryWriter

g_starttime = None
g_memstat = {}

g_devmode = True if int(os.environ.get('K12AI_DEV_MODE', '0')) else False
g_tbwriter = None
g_debug = False


def _get_writer():
    global g_tbwriter
    if g_devmode:
        if not g_tbwriter:
            logdir = '/cache/tblogs'
            # if os.path.exists(logdir):
            #     shutil.rmtree(logdir, ignore_errors=True)
            #     os.mkdir(logdir)
            g_tbwriter = SummaryWriter(log_dir=logdir)
    return g_tbwriter


def _tensor_to_list(x):
    return x.cpu().numpy().astype(float).reshape(-1).tolist()


def _except_message(exc_type=None, exc_value=None, exc_tb=None):
    if exc_type is None or exc_value is None or exc_tb is None:
        exc_type, exc_value, exc_tb = sys.exc_info()
        exc_type = exc_type.__name__
        exc_value = str(exc_value)

    message = {
        'err_type': exc_type,
        'err_text': exc_value
    }
    message['trackback'] = []
    tbs = traceback.extract_tb(exc_tb)
    for tb in tbs[1:]:
        err = {
            'filename': tb.filename,
            'linenum': tb.lineno,
            'funcname': tb.name,
            'source': tb.line
        }
        message['trackback'].append(err)
    return message


def _memstat_message():
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
    WARNING = 2
    ERROR = 3
    EXCEPT = 4
    STOP = 5
    FINISH = 6

    @staticmethod
    def logw(*args, **kwargs):
        print(*args)
        return MessageReport.status(MessageReport.WARNING)

    @staticmethod
    def status(what, msg=None):
        if what == MessageReport.RUNNING:
            global g_starttime
            g_starttime = time.time()
            k12ai_send_message('error', {
                'status': 'running',
                'memstat': _memstat_message()
            })
            return

        if what == MessageReport.WARNING:
            k12ai_send_message('error', {
                'status': 'warning',
                'errinfo': msg or _except_message()
            })
            return

        if what == MessageReport.ERROR:
            k12ai_send_message('error', {
                'status': 'error',
                'errinfo': msg or _except_message()
            })
            return

        if what == MessageReport.EXCEPT:
            k12ai_send_message('error', {
                'status': 'crash',
                'errinfo': msg or _except_message()
            })
            return

        if what == MessageReport.STOP:
            k12ai_send_message('error', {
                'status': 'stop',
                'event': msg or {'by comamnd way'}
            })
            sys.exit(0) # TODO
            return

        if what == MessageReport.FINISH:
            time.sleep(2) # TODO TDZ: database async, must write metrics message before status message
            k12ai_send_message('error', {
                'status': 'finish',
                'uptime': int(time.time() - g_starttime),
                'memstat': _memstat_message()
            })
            return

    @staticmethod
    def metrics(metrics, memstat=False, end=False):
        pass


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
                'title': f'{category}_{title}',
                'payload': payload,
            }
        }
        if width:
            obj['width'] = width
        if height:
            obj['height'] = height
        return obj

    @handle_exception(MessageReport.logw)
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
            raise NotImplementedError(type(x))
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
                    NotImplementedError(title)
            else:
                NotImplementedError(type(y))
        # DEV
        if title == 'progress' and self._writer is None:
            return self
        obj = self._mmjson('scalar', category, title, payload, width, height)
        self._metrics.append(obj)
        return self

    @handle_exception(MessageReport.logw)
    def add_image(self, category, title, image, fmt='base64string', step=None, width=None, height=None):
        if image is None:
            print('add_image value is None')
            return
        if fmt == 'base64string':
            imgbytes = image2bytes(image, width, height)
            payload = base64.b64encode(imgbytes).decode()
        elif fmt == 'svg':
            payload = image
            chklen = len(payload)
            if chklen > 200000:
                print("image length is too large: %d" % chklen)
        elif fmt == 'url':
            if image.startswith('/datasets'):
                payload = '/'.join(image.split('/')[3:])
            else:
                payload = image
        else:
            raise NotImplementedError(f'{fmt}')

        obj = self._mmjson('image', category, title, payload, width, height)
        obj['data']['format'] = fmt
        self._metrics.append(obj)
        if g_debug and fmt == 'base64string':
            with open(f'/cache/{obj["_id_"]}.png', 'wb') as fout:
                fout.write(base64.b64decode(payload))

        if self._writer:
            if fmt in ('url', 'svg'):
                imgbytes = image2bytes(image, width, height)
            tsave = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None
            self._writer.add_image(f'{category}/{title}',
                    numpy.asarray(Image.open(io.BytesIO(imgbytes))), step, dataformats='HWC')
            Image.MAX_IMAGE_PIXELS = tsave
        return self

    @handle_exception(MessageReport.logw)
    def add_matrix(self, category, title, value, step=None, width=None, height=None):
        if value is None:
            print('add_maxtrix value is None')
            return
        if self._writer:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(value, annot=True, fmt='d', linewidth=0.5, cmap='Blues', ax=ax, cbar=True)
            self._writer.add_figure(f'{category}/{title}', fig, step, close=False)
        if isinstance(value, numpy.ndarray):
            value = value.tolist()
        obj = self._mmjson('matrix', category, title, value, width, height)
        self._metrics.append(obj)
        return self

    @handle_exception(MessageReport.logw)
    def add_text(self, category, title, value, step=None, width=None, height=None):
        if value is None:
            print('add_text value is None')
            return
        obj = self._mmjson('text', category, title, value, width, height)
        self._metrics.append(obj)
        if self._writer:
            self._writer.add_text(f'{category}/{title}', f'{value}', step)
        return self

    @handle_exception(MessageReport.logw)
    def add_histogram(self, category, title, value, step=None, width=None, height=None):
        if value is None:
            print('add_histogram value is None')
            return
        if self._writer:
            self._writer.add_histogram(f'{category}/{title}', value, step)
        return self

    @handle_exception(MessageReport.logw)
    def add_graph(self, category, title, model, inputs, width=None, height=None):
        if model is None:
            print('add_graph value is None')
            return
        if self._writer:
            self._writer.add_graph(model, inputs)
        return self

    @handle_exception(MessageReport.logw)
    def add_video(self, category, title, value, fmt='base64string', step=None, width=None, height=None):
        if value is None:
            print('add_video value is None')
            return
        if isinstance(value, str):
            if os.path.getsize(value) > 2048000:
                return self

            if self._writer:
                import skvideo.io
                video = skvideo.io.vread(value)
                video = torch.Tensor(video).to(torch.uint8).unsqueeze(0).permute((0, 1, 4, 2, 3))
                self._writer.add_video(f'{category}/{title}', video, step, fps=60)

            video = io.open(value, 'r+b').read()
            payload = base64.b64encode(video).decode()
            obj = self._mmjson('video', category, title, payload, width=width, height=height)
            obj['data']['format'] = fmt
            self._metrics.append(obj)
        return self

    @handle_exception(MessageReport.logw)
    def add_embedding(self, category, title, value, metadata=None, step=None, width=None, height=None):
        if self._writer:
            self._writer.add_embedding(tag=f'{category}_{title}', mat=value, metadata=metadata, global_step=step)
        fig = dr_scatter2D(value, metadata)
        return self.add_image(category, title, fig, 'base64string', step, width, height)
