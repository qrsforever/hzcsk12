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

from k12ai.common.util_misc import sw_list


def k12ai_get_metrics(model, X_all, y_all, y_true, y_pred, kwargs):
    metrics = {}
    if 'r2' in kwargs:
        metrics['r2_score'] = sw_list(r2_score(y_true, y_pred, **kwargs['r2']))
    if 'mae' in kwargs:
        metrics['MAE'] = sw_list(mean_absolute_error(y_true, y_pred, **kwargs['mae']))
    if 'mse' in kwargs:
        metrics['MSE'] = sw_list(mean_squared_error(y_true, y_pred, **kwargs['mse']))
    if 'msle' in kwargs:
        metrics['MSLE'] = sw_list(mean_squared_log_error(y_true, y_pred, **kwargs['msle']))
    if 'mdae' in kwargs:
        metrics['MDAE'] = sw_list(median_absolute_error(y_true, y_pred, **kwargs['mdae']))
    if 'evs' in kwargs:
        metrics['EVS'] = sw_list(explained_variance_score(y_true, y_pred, **kwargs['evs']))
    return metrics
