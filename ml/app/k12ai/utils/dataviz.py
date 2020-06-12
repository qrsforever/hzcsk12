#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file dataviz.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-12 23:17


import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from k12ai.common.util_misc import plot_roc_curve


def make_roc(model, data, kwargs):
    classes = np.unique(data['y_test'])
    y_proba = model.predict_proba(data['X_test'])
    y_score = roc_auc_score(data['y_test'], y_proba, **kwargs)

    y_true = label_binarize(data['y_test'], classes=classes)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fig = plot_roc_curve(fpr, tpr, roc_auc)
    return y_score, fig
