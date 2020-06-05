#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file adaboost.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-05 16:02

from k12ai.models.base import K12Algorithm
from sklearn.ensemble import AdaBoostRegressor as Algo


class SKAdaBoostRegressor(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
