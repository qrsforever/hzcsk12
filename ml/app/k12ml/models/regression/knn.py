#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file knn.py
# @brief
# @author QRS
# @blog qrsforever.github.io
# @version 1.0
# @date 2020-02-13 15:02


from k12ml.models.base import K12Algorithm
from sklearn.neighbors import KNeighborsRegressor as Algo
from sklearn.metrics import r2_score


class SKKNN(K12Algorithm):
    def __init__(self, kwargs):
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

    def train(self, dataloader):
        X_train, X_test, Y_train, Y_test = dataloader.get_dataset()
        self.fit(X_train, Y_train)
        Y_prediction = self.predict(X_test)
        r2score = r2_score(Y_test, Y_prediction)
        return {
                'algorithm': self._algo.__class__.__name__,
                'r2_score': r2score
        }
