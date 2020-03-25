#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file clustering.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-27 10:42

import numpy as np

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import v_measure_score

from k12ai.common.util_misc import (sw_list, make_meshgrid, plot_decision_boundaries)
from k12ai.common.log_message import MessageMetric


def k12ai_get_metrics(model, X_all, y_all, y_true, y_pred, kwargs):
    mm = MessageMetric()

    # decision boundaries
    if X_all.shape[1] == 2:
        C0, C1 = model.centers[:, 0], model.centers[:, 1]
        X0, X1 = X_all[:, 0], X_all[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()])
        fig = plot_decision_boundaries(xx, yy, zz, X0, X1, y_all, C0, C1)
        mm.add_image('metrics', f'PCA-2D: {model.name}', fig)

    mm.send()

    metrics = {}
    if 'ami' in kwargs:
        metrics['AMI'] = sw_list(adjusted_mutual_info_score(y_true, y_pred, **kwargs['ami']))
    if 'ari' in kwargs:
        metrics['ARI'] = sw_list(adjusted_rand_score(y_true, y_pred))
    if 'fmi' in kwargs:
        metrics['FMI'] = sw_list(fowlkes_mallows_score(y_true, y_pred, **kwargs['fmi']))
    if 'completeness_score' in kwargs:
        metrics['completeness_score'] = sw_list(completeness_score(y_true, y_pred))
    if 'homogeneity_score' in kwargs:
        metrics['homogeneity_score'] = sw_list(homogeneity_score(y_true, y_pred))
    if 'v_measure_score' in kwargs:
        metrics['v_measure_score'] = sw_list(v_measure_score(y_true, y_pred, **kwargs['v_measure_score']))

    return metrics
