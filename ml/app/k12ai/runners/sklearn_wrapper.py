#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file sklearn_wrapper.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 21:01

from k12ai.runners.base import BaseRunner
from k12ai.data.data_loader import DataLoader


class SKRunner(BaseRunner):
    def __init__(self, configer):
        task = configer.get('task')
        if task == 'classifier':
            from k12ai.models.classification import k12ai_get_model
            from k12ai.metrics.sk.classification import k12ai_get_metrics
        elif task == 'regressor':
            from k12ai.models.regression import k12ai_get_model
            from k12ai.metrics.sk.regression import k12ai_get_metrics
        elif task == 'cluster':
            from k12ai.models.clustering import k12ai_get_model
            from k12ai.metrics.sk.clustering import k12ai_get_metrics
        else:
            raise NotImplementedError(f'task:{task}')

        model_name = configer.get('model.name')
        model_args = configer.get(f'model.{model_name}')

        self._model = k12ai_get_model(model_name)(**model_args)
        self._dataloader = DataLoader(configer)
        self._metrics = lambda data, \
                    y_true, y_pred, kwargs=configer.get('metrics'): \
                    k12ai_get_metrics(self._model, data, y_true, y_pred, kwargs)

    def train(self):
        data = self._dataloader.get_dataset()
        y_pred = self._model.train(data['X_train'], data['y_train'], data['X_test'])
        return self._metrics(data, data['y_test'], y_pred)
