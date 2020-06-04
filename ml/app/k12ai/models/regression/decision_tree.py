#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file decision_tree.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-13 18:29


from k12ai.models.base import K12Algorithm
from sklearn.tree import DecisionTreeRegressor as Algo


class SKDecisionTreeRegressor(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
