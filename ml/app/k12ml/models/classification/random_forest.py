#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file random_forest.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 18:42

from k12ml.models.base import K12Algorithm


class SKRandomForest(K12Algorithm):
    def __init__(self, configer):
        self._configer = configer
        self._algo = None

    def fit(self, X, Y):
        from sklearn.ensemble import RandomForestClassifier
        self._algo = RandomForestClassifier()
        self._algo.fit(X, Y)
        return self

    def predict(self, X):
        if self._algo is None:
            raise NotImplementedError
        return self._algo.predict(X)
