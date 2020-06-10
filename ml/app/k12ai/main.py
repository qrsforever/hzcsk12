#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 19:32

import os
import argparse
import datetime
from pyhocon import ConfigFactory
from k12ai.utils.logger import Logger
from k12ai.common.log_message import MessageReport
from k12ai.common.log_message import MessageMetric


def _do_train(configer):
    Logger.info(f'{configer}')
    mm = MessageMetric()
    if configer.get('method') == 'sklearn_wrapper':
        from k12ai.runners.sklearn_wrapper import SKRunner as Runner
    elif configer.get('method') == 'xgboost_wrapper':
        from k12ai.runners.xgboost_wrapper import XGBRunner as Runner
    else:
        raise NotImplementedError

    mm.add_text('train', 'remain_time', f'{datetime.timedelta(seconds=110)}').send()

    runner = Runner(configer)
    metrics = runner.train()
    # Text
    for key, value in metrics.items():
        mm.add_text('train', key, value)
    mm.send()
    Logger.info(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--phase',
            default=None,
            type=str,
            dest='phase',
            help="phase: train, evaluate")
    parser.add_argument(
            '--config_file',
            default=None,
            type=str,
            dest='config_file',
            help="configure file")
    args = parser.parse_args()

    try:
        MessageReport.status(MessageReport.RUNNING)
        if args.phase == 'train':
            if not os.path.exists(args.config_file):
                raise FileNotFoundError("file {} not found".format(args.config_file))
            _do_train(ConfigFactory.parse_file(args.config_file))
        else:
            raise NotImplementedError
        MessageReport.status(MessageReport.FINISH)
    except Exception:
        MessageReport.status(MessageReport.EXCEPT)
