#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file kmeans.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-14 23:17


from k12ai.models.base import K12Algorithm
from sklearn.cluster import KMeans as Algo
from sklearn.decomposition import PCA


class SKKmeans(K12Algorithm):
    def __init__(self, pca2=False, **kwargs):
        self._pca2 = pca2
        self._kwargs = kwargs
        self._algo = None

    def fit(self, X, Y):
        self._algo = Algo(**self._kwargs)
        if self._pca2:
            X = PCA(n_components=2).fit_transform(X)
        return self._algo.fit(X)

    def predict(self, X):
        if self._algo is None:
            raise NotImplementedError
        if self._pca2:
            X = PCA(n_components=2).fit_transform(X)
        return self._algo.predict(X)
