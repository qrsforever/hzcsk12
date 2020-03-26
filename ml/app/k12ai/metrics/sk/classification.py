#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file classification.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-27 10:42

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from k12ai.common.util_misc import (sw_list, make_meshgrid, plot_decision_boundaries)
from k12ai.common.log_message import MessageMetric as mm


def k12ai_get_metrics(model, data, y_true, y_pred, kwargs):

    # decision boundaries
    if data['X'].shape[1] == 2:
        X0, X1 = data['X'][:, 0], data['X'][:, 1]
        xx, yy = make_meshgrid(X0, X1)
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()])
        fig = plot_decision_boundaries(xx, yy, zz, X0, X1, data['y'])
        mm().add_image('DecisionBoundary', f'{model.name}', fig).send()

    # decision tree node
    if 'tree_dot' in kwargs:
        from sklearn.tree import export_graphviz
        import pydotplus
        try:
            tree_dot = export_graphviz(model.algo, feature_names=data['feature_names'])
            graph = pydotplus.graph_from_dot_data(tree_dot)
            mm().add_image('DecisionTree', f'{model.name}', graph.create_png()).send()
        except Exception as err:
            print(str(err))
            pass

    # text metrics
    metrics = {}
    if 'accuracy' in kwargs:
        metrics['accuracy_score'] = sw_list(accuracy_score(y_true, y_pred, **kwargs['accuracy']))
    if 'kappa' in kwargs:
        metrics['kappa_score'] = sw_list(cohen_kappa_score(y_true, y_pred, **kwargs['kappa']))
    if 'confusion_matrix' in kwargs:
        metrics['confusion_matrix'] = sw_list(confusion_matrix(y_true, y_pred, **kwargs['confusion_matrix']))
    if 'precision' in kwargs:
        metrics['precision_score'] = sw_list(precision_score(y_true, y_pred, **kwargs['precision']))
    if 'recall' in kwargs:
        metrics['recall_score'] = sw_list(recall_score(y_true, y_pred, **kwargs['recall']))
    if 'f1' in kwargs:
        metrics['f1_score'] = sw_list(f1_score(y_true, y_pred, **kwargs['f1']))
    if 'fbeta' in kwargs:
        metrics['fbeta_score'] = sw_list(fbeta_score(y_true, y_pred, **kwargs['fbeta']))
    if 'jaccard' in kwargs:
        metrics['jaccard_score'] = sw_list(jaccard_score(y_true, y_pred, **kwargs['jaccard']))
    if 'mcc' in kwargs:
        metrics['matthews_corrcoef'] = sw_list(matthews_corrcoef(y_true, y_pred, **kwargs['mcc']))
    return metrics
