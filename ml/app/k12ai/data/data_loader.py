#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file data_loader.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 21:30

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, configer):
        self._configer = configer

    def get_dataset(self):
        data_dir = self._configer.get('data.data_path')
        if data_dir.startswith('load_'):
            from k12ai.data.sklearn_dataset import sk_get_dataset as _get_dataset
        else:
            from k12ai.data.datasets import k12ai_get_dataset as _get_dataset

        # numpy array
        X, y, feature_names, target_names = _get_dataset(data_dir)

        pca2 = self._configer.get('data.pca2D', default=False)
        if pca2 and X.shape[1] > 2:
            # nomalize
            X = scale(X, copy=False)

            # pca 2D features
            X = PCA(n_components=2, copy=False).fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, **self._configer.get('data.sampling'))

        data = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'target_names': target_names,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        return data
