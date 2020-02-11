#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 19:51


class K12Algorithm:
    def __init__(self):
        pass

    def fit(self, X, Y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
