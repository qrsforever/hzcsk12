#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file sklearn_wrapper.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 21:01

from k12ml.runners.base import BaseRunner
from k12ml.data.data_loader import DataLoader


class SKRunner(BaseRunner):
    def __init__(self, model, configer):
        self._model = model
        self._dataloader = DataLoader(configer)

    def train(self):
        return self._model.train(self._dataloader)
