#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file gaussian_nb.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-13 22:43

from k12ai.models.base import K12Algorithm
from sklearn.naive_bayes import GaussianNB as Algo


class SKGaussianNB(K12Algorithm):
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
