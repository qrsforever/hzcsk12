#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file log_to_visdom.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-13 17:01:56

from typing import Set, Dict, TYPE_CHECKING
import logging

import time
import json
import torch

from allennlp.common.params import Params
from allennlp.training import util as training_util
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from allennlp.training.tensorboard_writer import TensorboardWriter
from visdom import Visdom
import GPUtil
import psutil

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer

logger = logging.getLogger(__name__)

@Callback.register("log_to_visdom")
class LogToVisdom(Callback):
    def __init__(self, server_port: int, min_interval: int = 5) -> None:
        logger.info("visdom server_port: %d", server_port)
        self.interval = min_interval
        self.port = server_port
        self.visdom = Visdom(port=server_port)
        self.widget = self.visdom.text("training_monitor")
        self.batch_start_time = 0.0
        self.last_report_time = 0
        self.monitor_stats = {
                "iters": 0,
                "lr": 0,
                "training_loss": 0,
                "training_speed": 0,
                "GPU": {},
                "CPU": {},
                "val_loss": 0,
                "acc": {},
                }

    @handle_event(Events.TRAINING_START)
    def training_start(self, trainer: 'CallbackTrainer'):
        logger.info("training start")

    @handle_event(Events.EPOCH_START)
    def epoch_start(self, trainer: 'CallbackTrainer'):
        logger.info("epoch start")

    @handle_event(Events.BATCH_START)
    def batch_start(self, trainer: 'CallbackTrainer'):
        self.batch_start_time = time.time()

    @handle_event(Events.BATCH_END)
    def batch_end(self, trainer: 'CallbackTrainer'):
        self.monitor_stats["iters"] += 1

        if time.time() - self.last_report_time < self.interval:
            return
        rate = 0
        for group in trainer.optimizer.param_groups:
            if 'lr' in group:
                rate = group['lr']
                break
        batch_elapsed_time = time.time() - self.batch_start_time
        self.monitor_stats["lr"] = rate
        self.monitor_stats["training_loss"] = trainer.train_metrics["loss"]
        self.monitor_stats["training_speed"] = 1 / batch_elapsed_time
        self.monitor_stats["CPU"] = {"CPU_Util": psutil.cpu_percent()}

        # TODO only one GPU
        gpu_info = GPUtil.getGPUs()[0]
        self.monitor_stats["GPU"] = {
                "GPU_Type": "{}".format(gpu_info.name),
                "GPU_Mem_Free": gpu_info.memoryFree,
                "GPU_Mem_Total": gpu_info.memoryTotal,
                "GPU_Util": gpu_info.load * 100,
                }

        self.visdom.text(text=json.dumps(self.monitor_stats), win=self.widget)
        self.last_report_time = time.time()

    @handle_event(Events.VALIDATE, priority=200)
    def validate(self, trainer: 'CallbackTrainer'):
        logger.info("validate")
        for key, value in trainer.val_metrics.items():
            if key == "accuracy":
                self.monitor_stats["acc"]["top1"] = value
            elif key == "accuracy3":
                self.monitor_stats["acc"]["top3"] = value
            elif key == "accuracy5":
                self.monitor_stats["acc"]["top5"] = value
        self.monitor_stats["val_loss"] = trainer.val_metrics["loss"]

    @handle_event(Events.EPOCH_END)
    def epoch_end(self, trainer: 'CallbackTrainer'):
        logger.info("epoch end")

    @handle_event(Events.TRAINING_END)
    def training_end(self, trainer: 'CallbackTrainer'):
        logger.info("trainning end")
