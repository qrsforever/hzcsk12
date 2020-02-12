#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file random_forest.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 18:42

from k12ml.models.base import K12Algorithm
from sklearn.metrics import accuracy_score


class SKRandomForest(K12Algorithm):
    def __init__(self, kwargs):
        self._kwargs = kwargs
        self._algo = None

    def fit(self, X, Y):
        from sklearn.ensemble import RandomForestClassifier
        self._algo = RandomForestClassifier(**self._kwargs)
        self._algo.fit(X, Y)
        return self

    def predict(self, X):
        if self._algo is None:
            raise NotImplementedError
        return self._algo.predict(X)

    def train(self, dataloader):
        X_train, X_test, Y_train, Y_test = dataloader.get_dataset()
        self.fit(X_train, Y_train)
        Y_prediction = self.predict(X_test)
        accs = accuracy_score(Y_test, Y_prediction)
        return {'accuracy_score': accs}
