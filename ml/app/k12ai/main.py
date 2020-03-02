#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 19:32

import os
import argparse
from pyhocon import ConfigFactory
from k12ai.utils.logger import Logger
from k12ai.common import k12ai_status_message


def _do_train(configer):
    Logger.info(f'{configer}')

    if configer.get('method') == 'sklearn_wrapper':
        from k12ai.runners.sklearn_wrapper import SKRunner
        runner = SKRunner(configer)
        metrics = runner.train()
        k12ai_status_message('k12ai_metrics', metrics)
        Logger.info(metrics)
    else:
        raise NotImplementedError


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
        k12ai_status_message('k12ai_running')
        if args.phase == 'train':
            if not os.path.exists(args.config_file):
                raise FileNotFoundError("file {} not found".format(args.config_file))
            _do_train(ConfigFactory.parse_file(args.config_file))
        else:
            raise NotImplementedError
        k12ai_status_message('k12ai_finish')
    except Exception:
        k12ai_status_message('k12ai_except')
