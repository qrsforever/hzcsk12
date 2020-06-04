#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file random_forest.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-13 15:53


from k12ai.models.base import K12Algorithm
from sklearn.ensemble import RandomForestRegressor as Algo


class SKRandomForestRegressor(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
