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


from k12ai.common.util_misc import (sw_list, plot_regression3D)
from k12ai.common.log_message import MessageMetric as mm


def k12ai_get_metrics(model, data, y_true, y_pred, kwargs):

    # 3D
    if data['X'].shape[1] == 2:
        X, Y = data['X_test'][:, 0], data['X_test'][:, 1]
        if len(y_true.shape) > 1:
            Z0 = y_true[:, 0]
            Z1 = y_pred[:, 0]
        else:
            Z0 = y_true
            Z1 = y_pred
        zlabels = data['target_names']
        if zlabels:
            zlabel = zlabels[0]
        else:
            zlabel = 'Z'
        fig = plot_regression3D(X, Y, Z0, Z1, zlabel=zlabel)
        mm().add_image('Regression3D', f'{model.name}', fig).send()

    # text metrics
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
    if 'auroc' in kwargs:
        from k12ai.utils.dataviz import make_roc
        score, fig = make_roc(model, data, kwargs['auroc'])
        mm().add_image('ROC', f'{model.name}', fig).send()
        metrics['auc'] = score
    return metrics
