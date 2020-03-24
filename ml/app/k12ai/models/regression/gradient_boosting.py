#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file gradient_boosting.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-26 22:45


from k12ai.models.base import K12Algorithm
from sklearn.ensemble import GradientBoostingRegressor as Algo


class SKGradientBoosting(K12Algorithm):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._algo = None

    def fit(self, X, Y):
        self._algo = Algo(**self._kwargs)
        self._algo.fit(X, Y)
        return self

    def predict(self, X):
        if self._algo is None:
            raise NotImplementedError
        return self._algo.predict(X)

    def train(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test)
