#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file kmeans.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-14 23:17


from k12ai.models.base import K12Algorithm
from sklearn.cluster import KMeans as Algo


class SKKmeans(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)

    @property
    def centers(self):
        return self._algo.cluster_centers_

    def fit(self, X, Y):
        return self.algo.fit(X)
