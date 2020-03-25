#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file decision_tree.py
# @brief
# @author QRS
# @blog qrsforever.github.io
# @version 1.0
# @date 2020-02-13 17:30

from k12ai.models.base import K12Algorithm
from sklearn.tree import DecisionTreeClassifier as Algo


class SKDecisionTree(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
