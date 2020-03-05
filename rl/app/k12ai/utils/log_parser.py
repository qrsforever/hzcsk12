#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_parser.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-22 15:19

import re
import numpy as np

from k12ai.common.log_message import MessageReport

g_phase = ''
g_iters = 0
g_metrics = {}


def k12ai_set_phase(phase):
    global g_phase 
    g_phase = phase


def _log_tabular(key, val):
    global g_metrics
    if g_iters > 0 and key not in (
            'Iteration', 'CumTime (s)', 'CumSteps',
            'GameScoreAverage', 'lossAverage'):
        return

    val = 0 if np.isnan(val) else val

    if g_phase == 'train':
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
            MessageReport.metrics(g_metrics)
    elif g_phase == 'evaluate':
        if key == "GameScoreAverage":
            g_metrics['evaluate_score'] = val
            g_metrics['evaluate_progress'] = 1.0
            MessageReport.metrics(g_metrics)
            return


def k12ai_log_parser(key_or_msg, val=None):
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
