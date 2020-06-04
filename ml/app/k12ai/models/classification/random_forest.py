#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file random_forest.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 18:42

from k12ai.models.base import K12Algorithm
from sklearn.ensemble import RandomForestClassifier as Algo


class SKRandomForestClassifier(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
