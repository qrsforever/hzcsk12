#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file xgboost_wrapper.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-10 20:49

from k12ai.runners.sklearn_wrapper import SKRunner


class XGBRunner(SKRunner):
    def __init__(self, configer):
        super().__init__(configer)
