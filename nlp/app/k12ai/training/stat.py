#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file stat.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-20 19:02

from k12ai.common.log_message import MessageMetric


class RunnerStat(object):
    @staticmethod
    def train(runner, metrics, **kwargs):
        epoch = metrics['epoch']
        mm = MessageMetric()
        # peak cpu/gpu
        if 'peak_cpu_memory_MB' in metrics:
            mm.add_scalar('train', 'cpu', x=epoch, y=metrics['peak_cpu_memory_MB'])
        y = {}
        for key, value in metrics.items():
            if key.startswith("peak_gpu_"):
                y[key] = value
        mm.add_scalar('train', 'gpu', x=epoch, y=y)

        # loss
        y = {}
        for key, value in metrics.items():
            if key.endswith('_loss'):
                y[key] = value
        mm.add_scalar('train_val', 'loss', x=epoch, y=y)

        # acc
        y = {}
        for key, value in metrics.items():
            if key.endswith('_accuracy'):
                y[key] = value
        mm.add_scalar('train_val', 'acc', x=epoch, y=y)

        mm.send()

    @staticmethod
    def validation(runner, *args, **kwargs):
        pass

    @staticmethod
    def evaluate(metrics, **kwargs):
        mm = MessageMetric()
        if 'loss' in metrics:
            mm.add_text('evaluate', 'loss', metrics['loss'])
        if 'accuracy' in metrics:
            mm.add_text('evaluate', 'acc', metrics['accuracy'])
        mm.send()
