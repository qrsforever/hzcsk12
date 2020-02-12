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
from k12ml.utils.logger import Logger
from k12ml.utils.rpc_message import hzcsk12_send_message as _sendmsg


def _do_train(configer):
    Logger.info(f'{configer}')
    task = configer.get('task')
    if task == 'classifier':
        from k12ml.models.classification import k12ai_get_model
        model_name = configer.get('model.name')
        model_algo = k12ai_get_model(model_name)(configer.get(f'model.{model_name}'))
    elif task == 'regressor':
        from k12ml.models.regression import k12ai_get_model
        model_name = configer.get('model.name')
        model_algo = k12ai_get_model(model_name)(configer.get(f'model.{model_name}'))
    else:
        raise NotImplementedError

    if configer.get('method') == 'sklearn_wrapper':
        from k12ml.runners.sklearn_wrapper import SKRunner
        runner = SKRunner(model_algo, configer)
        metrics = runner.train()
        _sendmsg(metrics)
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
        _sendmsg('k12ml_running')
        if args.phase == 'train':
            if not os.path.exists(args.config_file):
                raise FileNotFoundError("file {} not found".format(args.config_file))
            _do_train(ConfigFactory.parse_file(args.config_file))
        else:
            raise NotImplementedError
        _sendmsg('k12ml_finish')
    except Exception:
        _sendmsg('k12ml_except')
