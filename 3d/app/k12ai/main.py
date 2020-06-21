#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-21 20:30

import os
import argparse
import datetime
from pyhocon import ConfigFactory
from k12ai.utils.logger import Logger
from k12ai.common.log_message import MessageReport


def _do_train(configer):
    Logger.info(f'{configer}')
    # TODO


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
