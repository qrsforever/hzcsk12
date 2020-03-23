#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file clustering.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-27 10:42

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import v_measure_score
# from sklearn.preprocessing import scale

from k12ai.common.util_misc import sw_list


def k12ai_get_metrics(y_true, y_pred, kwargs):
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
