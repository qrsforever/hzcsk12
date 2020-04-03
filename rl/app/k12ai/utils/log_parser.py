#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_parser.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-22 15:19

import re
import numpy as np

from k12ai.common.log_message import MessageMetric

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
            'GameScoreAverage', 'lossAverage', 'ReturnAverage'):
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
        if key == "ReturnAverage":
            g_metrics['training_return'] = val
            return
        if key == "lossAverage":
            g_metrics['training_loss'] = val
            mm = MessageMetric()
            if 'training_loss' in g_metrics:
                mm.add_scalar('train', 'loss', x=g_metrics['training_iters'], y=g_metrics['training_loss']) 
            if 'training_score' in g_metrics:
                mm.add_scalar('train', 'score', x=g_metrics['training_iters'], y=g_metrics['training_score']) 
            if 'training_return' in g_metrics:
                mm.add_scalar('train', 'return', x=g_metrics['training_iters'], y=g_metrics['training_return']) 
            mm.send()
    elif g_phase == 'evaluate':
        if key == "Iteration":
            g_metrics = {}
            return
        if key == "GameScoreAverage":
            g_metrics['evaluate_score'] = val
            return
        if key == "ReturnAverage":
            g_metrics['evaluate_return'] = val
            return
        if key == "lossAverage": # end token
            g_metrics['evaluate_progress'] = 1.0
            mm = MessageMetric()
            if 'evaluate_score' in g_metrics:
                mm.add_text('evaluate', 'score', g_metrics['evaluate_score'])
            if 'evaluate_return' in g_metrics:
                mm.add_text('evaluate', 'return', g_metrics['evaluate_return']).send()
            mm.send()


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
