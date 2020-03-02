#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_parser.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 18:47:50

import sys
import re
import traceback
import resource
from torch import cuda

from k12ai.common import k12ai_send_message
from k12ai.common import k12ai_status_message

_isepoch_ = True
_maxiter_ = -1
_metrics_ = {}


def _parse_metrics(filename, lineno, message):
    global _isepoch_, _maxiter_ , _metrics_
    try:
        if filename == 'k12cv_init.py':
            if message.startswith('_k12ai.solver.lr.metric: '):
                res = re.search(r'_k12ai.solver.lr.metric: '
                        r'(?P<metric>[a-z]+), max: (?P<total>\d+)', message)
                if res:
                    result = res.groupdict()
                    if 'iters' == result.get('metric'):
                        _isepoch_ = False
                    _maxiter_ = int(result.get('total'))
        elif filename in ['image_classifier.py']:
            if message.startswith('Train Epoch:'):
                res = re.search(r'Train Epoch: (?P<epoch>\d+)\t'
                    r'Train Iteration: (?P<iters>\d+)\t'
                    r'Time (?P<batch_time_sum>\d+\.?\d*)s / (?P<batch_iters>\d+)iters, '
                    r'\((?P<batch_time_avg>\d+\.?\d*)\)\t'
                    r'Data load (?P<data_time_sum>\d+\.?\d*)s / (?P<_batch_iters>\d+)iters, '
                    r'\((?P<data_time_avg>\d+\.?\d*)\)\n'
                    r'Learning rate = (?P<learning_rate>.*)\t'
                    r'Loss = .*loss: (?P<train_loss>\d+\.?\d*).*\n', message)
                if res:
                    result = res.groupdict()
                    _metrics_ = {}
                    _metrics_['training_iters'] = int(result.get('iters', '0'))
                    _metrics_['training_epochs'] = int(result.get('epoch', '0'))
                    _metrics_['training_loss'] = float(result.get('train_loss', '0'))
                    _metrics_['training_speed'] = float(result.get('batch_time_avg', '0'))
                    _metrics_['lr'] = eval(result.get('learning_rate', '0'))
                    if _isepoch_:
                        _metrics_['training_progress'] = round(float(_metrics_['training_epochs']) / _maxiter_, 4)
                    else:
                        _metrics_['training_progress'] = round(float(_metrics_['training_iters']) / _maxiter_, 4)
            elif message.startswith('TestLoss = '):
                res = re.search(r'TestLoss = .*loss: (?P<val_loss>\d+\.?\d*).*', message)
                if res:
                    result = res.groupdict()
                    _metrics_['validation_loss'] = float(result.get('val_loss', '0'))
                return
            elif message.startswith('Top1 ACC = '):
                res = re.search(r'Top1 ACC = .*\'out\': (?P<acc>\d+\.?\d*).*', message)
                if res:
                    result = res.groupdict()
                    _metrics_['validation_accuracy'] = float(result.get('acc', '0'))
                return
            elif message.startswith('Top3 ACC = '):
                res = re.search(r'Top3 ACC = .*\'out\': (?P<acc>\d+\.?\d*).*', message)
                if res:
                    result = res.groupdict()
                    _metrics_['validation_accuracy3'] = float(result.get('acc', '0'))
                return
            elif message.startswith('Top5 ACC = '):
                res = re.search(r'Top5 ACC = .*\'out\': (?P<acc>\d+\.?\d*).*', message)
                if res:
                    result = res.groupdict()
                    _metrics_['validation_accuracy5'] = float(result.get('acc', '0'))
            else:
                return
        elif filename in ['faster_rcnn.py', 'single_shot_detector.py', 'yolov3.py']:
            if message.startswith('Train Epoch:'):
                res = re.search(r'Train Epoch: (?P<epoch>\d+)\t'
                    r'Train Iteration: (?P<iters>\d+)\t'
                    r'Time (?P<batch_time_sum>\d+\.?\d*)s / (?P<batch_iters>\d+)iters, '
                    r'\((?P<batch_time_avg>\d+\.?\d*)\)\t'
                    r'Data load (?P<data_time_sum>\d+\.?\d*)s / (?P<_batch_iters>\d+)iters, '
                    r'\((?P<data_time_avg>\d+\.?\d*)\)\n'
                    r'Learning rate = (?P<learning_rate>.*)\t'
                    r'Loss = (?P<train_loss>\d+\.?\d*) \(ave = (?P<loss_avg>\d+\.?\d*)\)\n', message)
                if res:
                    result = res.groupdict()
                    _metrics_ = {}
                    _metrics_['training_iters'] = int(result.get('iters', '0'))
                    _metrics_['training_epochs'] = int(result.get('epoch', '0'))
                    _metrics_['training_loss'] = float(result.get('train_loss', '0'))
                    _metrics_['training_speed'] = float(result.get('batch_time_avg', '0'))
                    _metrics_['lr'] = eval(result.get('learning_rate', '0'))
                    if _isepoch_:
                        _metrics_['training_progress'] = round(float(_metrics_['training_epochs']) / _maxiter_, 4)
                    else:
                        _metrics_['training_progress'] = round(float(_metrics_['training_iters']) / _maxiter_, 4)
            elif message.startswith('Test Time'):
                res = re.search(r'Test Time (?P<batch_time_sum>\d+\.?\d*)s, '
                        r'\((?P<batch_time_avg>\d+\.?\d*)\)\t'
                        r'Loss (?P<loss_avg>\d+\.?\d*)\n', message)
                if res:
                    result = res.groupdict()
                    _metrics_['validation_loss'] = float(result.get('loss_avg', '0'))
            elif message.startswith('Val mAP:'):
                res = re.search(r'Val mAP: (?P<mAP>\d+\.?\d*)', message)
                if res:
                    result = res.groupdict()
                    _metrics_['validation_mAP'] = round(float(result.get('mAP', '0')), 4)
            else:
                return
        elif filename == 'single_shot_detector_test.py':
            if message.startswith('Test mAP:'):
                res = re.search(r'Test mAP: (?P<mAP>\d+\.?\d*)', message)
                if res:
                    result = res.groupdict()
                    _metrics_['evaluate_mAP'] = round(float(result.get('mAP', '0')), 4)
                    _metrics_['evaluate_progress'] = 1.0
        else:
            return
        # send message to k12cv service
        k12ai_send_message('metrics', _metrics_)
    except Exception as err:
        print(err)


def _parse_error(filename, lineno, message):

    def _err_msg(err_type):
        _message = {}
        _message['filename'] = filename
        _message['linenum'] = lineno
        _message['err_type'] = err_type
        _message['err_text'] = message
        return k12ai_status_message('k12ai_error', _message)

    if message == 'Image type is invalid.':
        return _err_msg('ImageTypeError')

    if message == 'Tensor size is not valid.':
        return _err_msg('TensorSizeError')

    if re.search(r'Method: \w+ is not valid.', message):
        return _err_msg('ConfigurationError')

    if re.search(r'Not support BN type: \w+.', message):
        return _err_msg('ConfigurationError')

    if re.search(r'Model: \w+ not valid!', message):
        return _err_msg('InvalidModel')

    if re.search(r'Optimizer \w+ is not valid.', message):
        return _err_msg('InvalidOptimizerMethod')

    if re.search(r'Policy:\w+ is not valid.', message):
        return _err_msg('ConfigurationError')

    if re.search(r'Not support \w+ image tool.', message):
        return _err_msg('ConfigurationError')

    if re.search(r'Not support mode \w+', message):
        return _err_msg('ConfigurationError')

    if re.search(r'Invalid pad mode: \w+', message):
        return _err_msg('InvalidPadMode')

    if re.search(r'Anchor Method \w+ not valid.', message):
        return _err_msg('InvalidAnchorMethod')

    return _err_msg('UnkownError')


def _parse_except(filename, lineno, message):
    return k12ai_status_message('k12ai_except')


def k12ai_log_parser(level, filename, lineno, message):
    if level == 'info':
        _parse_metrics(filename, lineno, message)
    elif level == 'error':
        _parse_error(filename, lineno, message)
    elif level == 'critical':
        _parse_except(filename, lineno, message)
    else:
        print('Not impl yet!')
