#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_parser.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-22 15:19

import datetime
import numpy as np

from k12ai.common.log_message import MessageMetric as MM

g_max_steps = 0
g_cum_time = 0
g_loss_flag = False


def set_n_steps(n_steps):
    global g_max_steps
    g_max_steps = n_steps


def _log_tabular(key, val, itr):
    global g_cum_time, g_loss_flag
    if np.isnan(val):
        return

    if g_loss_flag:
        if key == 'CumTime (s)':
            g_cum_time = float(val)
            return

        if key == 'CumSteps':
            remain_time = g_cum_time * (g_max_steps / float(val) - 1)
            formatted_time = str(datetime.timedelta(seconds=int(remain_time)))
            MM().add_text('train', 'remain_time', f'{formatted_time}').send()
            return

    if key in ('StepsPerSecond', 'UpdatesPerSecond', 'ReplayRatio'):
        MM().add_scalar('Diagnostics', key, itr, val).send()
        return

    if key in ('GameScoreAverage', 'ReturnAverage', 'gradNormAverage',
            'NonzeroRewardsAverage', 'DiscountedReturnAverage'):
        MM().add_scalar('TrajInfos', key, itr, val).send()
        return

    if key in ('lossAverage', 'gradNormAverage'):
        MM().add_scalar('OptInfos', key, itr, val).send()
        if not g_loss_flag:
            g_loss_flag = True
        return


def k12ai_log_parser(key, val=None, itr=None):
    if val: # value
        return _log_tabular(key, val, itr)
