#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file kmeans.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-14 23:17


from k12ml.models.base import K12Algorithm
from sklearn.cluster import KMeans as Algo


class SKKmeans(K12Algorithm):
    def __init__(self, kwargs):
        self._kwargs = kwargs
        self._algo = None

    def fit(self, X, Y):
        self._algo = Algo(**self._kwargs)
        return self._algo.fit(X)

    def predict(self, X):
        if self._algo is None:
            raise NotImplementedError
        return self._algo.predict(X)

    def train(self, dataloader):
        X_train, X_test, Y_train, Y_test = dataloader.get_dataset()
        self.fit(X_train, Y_train)
        self.predict(X_test)
        return {
                'algorithm': self._algo.__class__.__name__,
        }
