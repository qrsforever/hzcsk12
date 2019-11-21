#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file handle_errors.py
# @brief
# @author QRS
# @blog qrsforever.github.io
# @version 1.0
# @date 2019-11-21 19:37:18

from typing import TYPE_CHECKING
import logging

from allennlp.common.util import hzcsk12_send_message
from allennlp.training.callbacks.callback import Callback, handle_event

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer

logger = logging.getLogger(__name__)

@Callback.register("k12errors")
class K12HandleErrors(Callback):

    @handle_event(Events.Error)
    def handle_errors(self, trainer: 'CallbackTrainer'):
        logger.error("handle error")
        self.exc = trainer.exception
        error  = {}
        error['class'] = trainer.exception.__class__
        error['message'] = str(trainer.exception)
        error['traceback'] = 'not impl'
        hzcsk12_send_message(msgtype='error', error, True)
