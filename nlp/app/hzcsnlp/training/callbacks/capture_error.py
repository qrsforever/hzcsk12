#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file capture_error.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-15 14:50:25

# from typing import Set, Dict, TYPE_CHECKING
# import logging
# 
# import time
# import json
# import torch
# 
# from allennlp.common.params import Params
# from allennlp.training import util as training_util
# from allennlp.training.callbacks.callback import Callback, handle_event
# 
# if TYPE_CHECKING:
#     from allennlp.training.callback_trainer import CallbackTrainer
# 
# logger = logging.getLogger(__name__)
# 
# @Callback.register("error")
# class CaptureError(Callback):
# 
#     @handle_event(Events.Error)
#     def capture_error(self, trainer: 'CallbackTrainer'):
#         logger.info("capture error")
#         self.exc = trainer.exception
