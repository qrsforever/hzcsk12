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
    def __init__(self, configer):
        task = configer.get('task')
        if task == 'classifier':
            from k12ml.models.classification import k12ai_get_model
        elif task == 'regressor':
            from k12ml.models.regression import k12ai_get_model
        elif task == 'cluster':
            from k12ml.models.clustering import k12ai_get_model
        else:
            raise NotImplementedError

        model_name = configer.get('model.name')
        model_args = configer.get(f'model.{model_name}')

        self._model = k12ai_get_model(model_name)(model_args)
        self._dataloader = DataLoader(configer)

    def train(self):
        return self._model.train(self._dataloader)
