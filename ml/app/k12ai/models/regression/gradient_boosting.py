#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file gradient_boosting.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-26 22:45


from k12ai.models.base import K12Algorithm
from sklearn.ensemble import GradientBoostingRegressor as Algo


class SKGradientBoosting(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
