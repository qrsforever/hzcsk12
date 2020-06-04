#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file knn.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 18:43

from k12ai.models.base import K12Algorithm
from sklearn.neighbors import KNeighborsClassifier as Algo


class SKKNeighborsClassifier(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
