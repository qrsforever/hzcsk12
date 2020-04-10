#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_parser.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-22 15:19

import re
import numpy as np

from k12ai.common.log_message import MessageMetric as MM

g_max_iters = 0


def _log_tabular(key, val, itr):
    if np.isnan(val):
        return

    if key == 'Iteration':
        MM().add_scalar('k12ai', 'progress', itr, round(100 * itr / g_max_iters, 2)).send()
        return

    if key in ('StepsPerSecond', 'UpdatesPerSecond', 'ReplayRatio'):
        MM().add_scalar('Diagnostics', key, itr, val).send()

    if key in ('GameScoreAverage', 'ReturnAverage', 'gradNormAverage',
            'NonzeroRewardsAverage', 'DiscountedReturnAverage'):
        MM().add_scalar('TrajInfos', key, itr, val).send()

    if key in ('lossAverage', 'gradNormAverage'):
        MM().add_scalar('OptInfos', key, itr, val).send()


def k12ai_log_parser(key, val=None, itr=None):
    if val: # value
        return _log_tabular(key, val, itr)

    global g_max_iters
    if g_max_iters == 0 and key.startswith('Running '):
        result = re.search(r'\ARunning (?P<iters>\d+) iterations of minibatch RL.', key)
        if not result:
            result = re.search(r'\ARunning (?P<iters>\d+) sampler iterations.', key)
        if result:
            g_max_iters = int(result.groupdict().get('iters', 0))
