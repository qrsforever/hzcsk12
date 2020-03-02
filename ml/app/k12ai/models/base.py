#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 19:51


class K12Algorithm:
    def fit(self, X, Y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def train(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test)
