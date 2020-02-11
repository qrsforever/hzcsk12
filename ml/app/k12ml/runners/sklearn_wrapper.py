#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file sklearn_wrapper.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 21:01

from k12ml.runners.base import BaseRunner
from k12ml.data.data_loader import DataLoader
from sklearn.metrics import accuracy_score


class SKRunner(BaseRunner):
    def __init__(self, model, configer):
        self._model = model
        self._configer = configer

        dataloader = DataLoader(configer)
        X_train, X_test, Y_train, Y_test = dataloader.get_dataset()
        self._X_train = X_train
        self._X_test = X_test
        self._Y_train = Y_train
        self._Y_test = Y_test

    def fit(self):
        return self._model.fit(self._X_train, self._Y_train)

    def predict(self):
        Y_predictions = self._model.predict(self._X_test)
        print("accuracy score: ", accuracy_score(self._Y_test, Y_predictions))
