#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 19:51


class K12Algorithm:
    def __init__(self, Algo, *arg, **kwargs):
        self._algo = Algo(**kwargs)

    @property
    def algo(self):
        return self._algo

    @property
    def name(self):
        return self.algo.__class__.__name__

    def fit(self, X, Y):
        return self.algo.fit(X, Y)

    def predict(self, X):
        return self.algo.predict(X)

    def predict_proba(self, X):
        return self.algo.predict_proba(X)

    def train(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test)
