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
