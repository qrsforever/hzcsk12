#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_parser.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-22 15:19

import os, sys
import re
import traceback
import numpy as np

from k12rl.utils.rpc_message import hzcsk12_send_message
from k12rl.utils import hzcsk12_kill

g_iters = 0
g_metrics = {}


def _log_tabular(key, val):
    global g_metrics
    if g_iters > 0 and key not in (
            'Iteration', 'CumTime (s)', 'CumSteps',
            'GameScoreAverage', 'lossAverage'):
        return

    val = 0 if np.isnan(val) else val

    if key == "Iteration":
        g_metrics = {}
        g_metrics['training_iters'] = val
        g_metrics['training_progress'] = round(val / g_iters, 4)
        return
    if key == "CumTime (s)":
        g_metrics['training_speed'] = val
        return
    if key == "CumSteps":
        g_metrics['training_speed'] = round(val / g_metrics['training_speed'], 4)
        return
    if key == "GameScoreAverage":
        g_metrics['training_score'] = val
        return
    if key == "lossAverage":
        g_metrics['training_loss'] = val
        hzcsk12_send_message('metrics', g_metrics)


def hzcsk12_log_message(errmsg):
    if errmsg.startswith('k12rl_running'):
        hzcsk12_send_message('status', {'value': 'running'})
        return

    if errmsg.startswith('k12rl_finish'):
        hzcsk12_send_message('status', {'value': 'exit', 'way': 'finish'})
        return

    if errmsg.startswith('k12rl_signal'):
        message = {'err_type': 'signal', 'err_text': 'handle quit signal'}
        hzcsk12_send_message('error', message)
        hzcsk12_send_message('status', {'value': 'exit', 'way': 'error'})
        hzcsk12_kill(os.getpid())
        return

    if errmsg.startswith('k12rl_except'):
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        exc_type, exc_value, exc_tb = sys.exc_info()
        message = {
            'filename': filename,
            'linenum': lineno,
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
                'souce': tb.line
            }
            message['trackback'].append(err)
        print(message)
        hzcsk12_send_message('error', message)
        hzcsk12_send_message('status', {'value': 'exit', 'way': 'crash'})
        hzcsk12_kill(os.getpid())


def hzcsk12_log_parser(key_or_msg, val=None):
    if val:
        return _log_tabular(key_or_msg, val)

    message = key_or_msg
    global g_iters
    if g_iters == 0 and message.startswith('Running '):
        result = re.search(r'\ARunning (?P<iters>\d+) iterations of minibatch RL.', message)
        if not result:
            result = re.search(r'\ARunning (?P<iters>\d+) sampler iterations.', message)
        if result:
            g_iters = int(result.groupdict().get('iters', 0))
