#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file svc.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 18:33

from k12ml.models.base import K12Algorithm
from sklearn.metrics import r2_score


class SKSVR(K12Algorithm):
    def __init__(self, kwargs):
        self._kwargs = kwargs
        self._algo = None

    def fit(self, X, Y):
        from sklearn.svm import SVR
        self._algo = SVR(**self._kwargs)
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
        r2score = r2_score(Y_test, Y_prediction)
        return {'r2_score': r2score}
