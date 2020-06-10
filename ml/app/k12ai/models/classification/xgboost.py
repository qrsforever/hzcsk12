#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file xgboost.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-10 20:56


from k12ai.models.base import K12Algorithm
from xgboost import XGBClassifier as Algo


class XGBoostClassifier(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
