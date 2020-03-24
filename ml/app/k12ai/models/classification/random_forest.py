#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file random_forest.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 18:42

from k12ai.models.base import K12Algorithm
from sklearn.ensemble import RandomForestClassifier as Algo


class SKRandomForest(K12Algorithm):
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
