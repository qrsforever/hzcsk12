#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file svc.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 18:33

from k12ml.models.base import K12Algorithm


class SKSVC(K12Algorithm):
    def __init__(self, kwargs):
        self._kwargs = kwargs
        self._algo = None

    def fit(self, X, Y):
        from sklearn.svm import SVC
        self._algo = SVC(**self._kwargs)
        self._algo.fit(X, Y)
        return self

    def predict(self, X):
        if self._algo is None:
            raise NotImplementedError
        return self._algo.predict(X)
