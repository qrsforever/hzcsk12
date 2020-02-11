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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--phase',
            default=None,
            type=str,
            dest='phase',
            help="phase: fit, predict")
    parser.add_argument(
            '--config_file',
            default=None,
            type=str,
            dest='config_file',
            help="configure file")
    args = parser.parse_args()

    if args.phase == 'fit':
        if not os.path.exists(args.config_file):
            Logger.error('config file {} is not exists!'.format(args.config_file))
            exit(1)

        configer = ConfigFactory.parse_file(args.config_file)
        if configer.get('task') == 'classifier':
            from k12ml.models.classification import k12ai_get_model
            model = k12ai_get_model(configer.get('model.name'))(configer.get('model.args'))
            if configer.get('method') == 'sklearn_wrapper':
                from k12ml.runners.sklearn_wrapper import SKRunner
                runner = SKRunner(model, configer)
                runner.fit()
                runner.predict()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
