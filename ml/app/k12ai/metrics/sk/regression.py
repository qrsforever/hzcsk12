#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file regression.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-27 10:29

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

import numpy as np


def _sw(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def k12ai_get_metrics(y_true, y_pred, kwargs):
    metrics = {}
    if 'r2' in kwargs:
        metrics['r2_score'] = _sw(r2_score(y_true, y_pred, **kwargs['r2']))
    if 'mae' in kwargs:
        metrics['MAE'] = _sw(mean_absolute_error(y_true, y_pred, **kwargs['mae']))
    if 'mse' in kwargs:
        metrics['MSE'] = _sw(mean_squared_error(y_true, y_pred, **kwargs['mse']))
    if 'msle' in kwargs:
        metrics['MSLE'] = _sw(mean_squared_log_error(y_true, y_pred, **kwargs['msle']))
    if 'mdae' in kwargs:
        metrics['MDAE'] = _sw(median_absolute_error(y_true, y_pred, **kwargs['mdae']))
    if 'evs' in kwargs:
        metrics['EVS'] = _sw(explained_variance_score(y_true, y_pred, **kwargs['evs']))
    return metrics
