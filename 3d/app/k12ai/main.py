#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-21 20:30

import os
import argparse
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter
from k12ai.utils.logger import Logger
from k12ai.common.log_message import MessageReport
from k12ai.data.dataloader import Dataloader
from k12ai.modules.criterion import CRITERION_DICT
from k12ai.modules.optimizer import OPTIMIZER_DICT
from k12ai.modules.scheduler import SCHEDULER_DICT


def _do_train(configer):
    Logger.info(HOCONConverter.convert(configer, 'json'))
    dataloader = Dataloader(configer.get('data'))
    train_dataloader = dataloader.get_trainloader()
    valid_dataloader = dataloader.get_validloader()
    if configer.get('model.network') == 'fcrn':
        from k12ai.models.fcrn import ResNet
        from k12ai.runners.trainer import DepthPredictTrainer as Trainer
        model = ResNet().cuda()
        criterion = CRITERION_DICT[configer.get('hypes.criterion.type')](**configer.get('hypes.criterion.args', default={}))
        optimizer = OPTIMIZER_DICT[configer.get('hypes.optimizer.type')](model.parameters(), **configer.get('hypes.optimizer.args', default={}))
        scheduler = SCHEDULER_DICT[configer.get('hypes.scheduler.type')](optimizer, **configer.get('hypes.scheduler.args', default={}))
        trainer = Trainer(
                train_dataloader,
                valid_dataloader,
                model, criterion, optimizer, scheduler,
                configer.get('hypes.epoch'), configer.get('metrics'),
                cache_dir='/cache/output')
    else:
        raise NotImplementedError

    trainer.train()


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
        elif args.phase == 'evaluate':
            raise NotImplementedError
        else:
            raise NotImplementedError
        MessageReport.status(MessageReport.FINISH)
    except Exception:
        MessageReport.status(MessageReport.EXCEPT)
