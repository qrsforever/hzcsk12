#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file logistic.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-24 22:52


from k12ai.models.base import K12Algorithm
from sklearn.linear_model import LogisticRegression as Algo


class SKLogisticRegression(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
