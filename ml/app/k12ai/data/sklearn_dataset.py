#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file sklearn_dataset.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 20:41

from sklearn.model_selection import train_test_split
import sklearn.datasets


def sk_get_dataset(dataset, kwargs):
    data = getattr(sklearn.datasets, dataset)()
    return train_test_split(data.data, data.target, **kwargs)
