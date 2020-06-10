#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file xgboost.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-10 20:57


from k12ai.models.base import K12Algorithm
from xgboost import XGBRegressor as Algo


class XGBoostRegressor(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
