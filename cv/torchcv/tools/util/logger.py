#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Logging tool implemented with the python Package logging.


import argparse
import logging
import os
import sys

# QRS: add
import re
import zerorpc

_RPCClient = None
_RPCEnable = -1
K12CV_TASK, K12CV_USER, K12CV_UUID = None, None, None

def hzcsk12_send_message(msgtype, message, end=False):
    global _RPCClient, _RPCEnable, K12CV_TASK, K12CV_USER, K12CV_UUID

    if _RPCEnable == 0:
        return

    if _RPCEnable == -1:
        host = os.environ.get('K12CV_RPC_HOST', None)
        port = os.environ.get('K12CV_RPC_PORT', None)
        print(host, port)
        if not host or not port:
            _RPCEnable = 0
            return
        K12CV_TASK = os.environ.get('K12CV_TASK', 'Unkown')
        K12CV_USER = os.environ.get('K12CV_USER', 'Unkown')
        K12CV_UUID = os.environ.get('K12CV_UUID', 'Unkown')
        _RPCClient = zerorpc.Client(
                connect_to='tcp://{}:{}'.format(host, port),
                timeout=2,
                passive_heartbeat=True)
        _RPCEnable = 1

    try:
        if message:
            _RPCClient.send_message(K12CV_TASK, K12CV_USER, K12CV_UUID, msgtype, message)
        if end:
            _RPCClient.close()
    except Exception:
        pass

_metrics_ = {}

RE_CLS_IC_TRAIN=None
RE_DET_COM_TRAIN=None

def hzcsk12_log_message(filename, message):
    global _metrics_
    try:
        if filename == 'image_classifier.py':
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
        if filename in ['faster_rcnn.py', 'single_shot_detector.py', 'yolov3.py']:
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
                        r'Loss (?P<loss_avg>\d+\.?\d*)\n')
                if res:
                    result = res.groupdict()
                    _metrics_['validation_loss'] = float(result.get('loss_avg', '0'))
            elif message.startwith('Val mAP:'):
                res = re.search(r'Val mAP: (?P<mAP>\d+\.?\d*)')
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


DEFAULT_LOG_LEVEL = 'info'
DEFAULT_LOG_FORMAT = '%(asctime)s %(levelname)-7s %(message)s'

LOG_LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


class Logger(object):
    """
    Args:
      Log level: CRITICAL>ERROR>WARNING>INFO>DEBUG.
      log format: The format of log messages.
    """
    logger = None

    @staticmethod
    def init(log_format=DEFAULT_LOG_FORMAT,
             log_level=DEFAULT_LOG_LEVEL,
             distributed_rank=0):
        assert Logger.logger is None
        Logger.logger = logging.getLogger()
        if distributed_rank > 0:
            return

        if log_level not in LOG_LEVEL_DICT:
            print('Invalid logging level: {}'.format(log_level))
            return

        Logger.logger.setLevel(LOG_LEVEL_DICT[log_level])
        fmt = logging.Formatter(log_format)
        console = logging.StreamHandler()
        console.setLevel(LOG_LEVEL_DICT[log_level])
        console.setFormatter(fmt)
        Logger.logger.addHandler(console)

    @staticmethod
    def check_logger():
        if Logger.logger is None:
            Logger.init(log_level=DEFAULT_LOG_LEVEL, log_format=DEFAULT_LOG_FORMAT)

    @staticmethod
    def debug(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        Logger.logger.debug('{} {}'.format(prefix, message))

    @staticmethod
    def info(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        hzcsk12_log_message(filename, message)
        prefix = '[{}, {}]'.format(filename,lineno)
        Logger.logger.info('{} {}'.format(prefix, message))

    @staticmethod
    def warn(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        Logger.logger.warn('{} {}'.format(prefix, message))

    @staticmethod
    def error(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        Logger.logger.error('{} {}'.format(prefix, message))

    @staticmethod
    def critical(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        Logger.logger.critical('{} {}'.format(prefix, message))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', default=None, type=str,
                        dest='log_level', help='To set the level to print to screen.')
    parser.add_argument('--log_format', default="%(asctime)s %(levelname)-7s %(message)s",
                        type=str, dest='log_format', help='The format of log messages.')

    args = parser.parse_args()
    Logger.init(log_level=args.log_level, log_format=args.log_format)

    Logger.info("info test.")
    Logger.debug("debug test.")
    Logger.warn("warn test.")
    Logger.error("error test.")
    Logger.debug("debug test.")
