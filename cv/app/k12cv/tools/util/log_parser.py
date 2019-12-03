#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_parser.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 18:47:50

import re
from k12cv.tools.util.rpc_message import hzcsk12_send_message

_metrics_ = {}

RE_CLS_IC_TRAIN = None
RE_DET_COM_TRAIN = None

def hzcsk12_log_parser(filename, message):
    global _metrics_
    try:
        if filename in ['image_classifier.py']:
            if message.startswith('Train Epoch:'):
                global RE_CLS_IC_TRAIN
                if RE_CLS_IC_TRAIN is None:
                    RE_CLS_IC_TRAIN = re.compile(r'Train Epoch: (?P<epoch>\d+)\t'
                            r'Train Iteration: (?P<iters>\d+)\t'
                            r'Time (?P<batch_time_sum>\d+\.?\d*)s / (?P<batch_iters>\d+)iters, '
                            r'\((?P<batch_time_avg>\d+\.?\d*)\)\t'
                            r'Data load (?P<data_time_sum>\d+\.?\d*)s / (?P<_batch_iters>\d+)iters, '
                            r'\((?P<data_time_avg>\d+\.?\d*)\)\n'
                            r'Learning rate = (?P<learning_rate>.*)\t'
                            r'Loss = .*loss: (?P<train_loss>\d+\.?\d*).*\n')
                res = RE_CLS_IC_TRAIN.search(message)
                if res:
                    result = res.groupdict()
                    _metrics_['training_epochs'] = int(result.get('epoch', '0'))
                    _metrics_['training_loss'] =  float(result.get('train_loss', '0'))
                    _metrics_['training_speed'] = float(result.get('batch_time_avg', '0'))
                    _metrics_['lr'] = eval(result.get('learning_rate', '0'))
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
                global RE_DET_COM_TRAIN
                if RE_DET_COM_TRAIN is None:
                    RE_DET_COM_TRAIN = re.compile(r'Train Epoch: (?P<epoch>\d+)\t'
                            r'Train Iteration: (?P<iters>\d+)\t'
                            r'Time (?P<batch_time_sum>\d+\.?\d*)s / (?P<batch_iters>\d+)iters, '
                            r'\((?P<batch_time_avg>\d+\.?\d*)\)\t'
                            r'Data load (?P<data_time_sum>\d+\.?\d*)s / (?P<_batch_iters>\d+)iters, '
                            r'\((?P<data_time_avg>\d+\.?\d*)\)\n'
                            r'Learning rate = (?P<learning_rate>.*)\t'
                            r'Loss = (?P<train_loss>\d+\.?\d*) \(ave = (?P<loss_avg>\d+\.?\d*)\)\n')
                res = RE_DET_COM_TRAIN.search(message)
                if res:
                    result = res.groupdict()
                    _metrics_['training_epochs'] = int(result.get('epoch', '0'))
                    _metrics_['training_loss'] =  float(result.get('train_loss', '0'))
                    _metrics_['training_speed'] = float(result.get('batch_time_avg', '0'))
                    _metrics_['lr'] = eval(result.get('learning_rate', '0'))
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
                    _metrics_['validation_mAP'] = float(result.get('mAP', '0'))
            else:
                return
        else:
            return
        # send message to k12cv service
        hzcsk12_send_message('metrics', _metrics_)
    except Exception as err:
        print(err)
