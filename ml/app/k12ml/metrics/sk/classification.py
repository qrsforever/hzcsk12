#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file classification.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-27 10:42

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import numpy as np


def _sw(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def k12ai_get_metrics(y_true, y_pred, kwargs):
    metrics = {}
    if 'accuracy' in kwargs:
        metrics['accuracy_score'] = _sw(accuracy_score(y_true, y_pred, **kwargs['accuracy']))
    if 'kappa' in kwargs:
        metrics['kappa_score'] = _sw(cohen_kappa_score(y_true, y_pred, **kwargs['kappa']))
    if 'confusion_matrix' in kwargs:
        metrics['confusion_matrix'] = _sw(confusion_matrix(y_true, y_pred, **kwargs['confusion_matrix']))
    if 'precision' in kwargs:
        metrics['precision_score'] = _sw(precision_score(y_true, y_pred, **kwargs['precision']))
    if 'recall' in kwargs:
        metrics['recall_score'] = _sw(recall_score(y_true, y_pred, **kwargs['recall']))
    if 'f1' in kwargs:
        metrics['f1_score'] = _sw(f1_score(y_true, y_pred, **kwargs['f1']))
    if 'fbeta' in kwargs:
        metrics['fbeta_score'] = _sw(fbeta_score(y_true, y_pred, **kwargs['fbeta']))
    if 'jaccard' in kwargs:
        metrics['jaccard_score'] = _sw(jaccard_score(y_true, y_pred, **kwargs['jaccard']))
    if 'mcc' in kwargs:
        metrics['matthews_corrcoef'] = _sw(matthews_corrcoef(y_true, y_pred, **kwargs['mcc']))
    return metrics
