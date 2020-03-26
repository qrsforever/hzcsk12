#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file sklearn_dataset.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 20:41

import sklearn.datasets


def sk_get_dataset(dataset):
    data = getattr(sklearn.datasets, dataset)()
    if hasattr(data, 'target_names'):
        target_names = data.target_names
    else:
        target_names = None
    return data.data, data.target, data.feature_names, target_names
