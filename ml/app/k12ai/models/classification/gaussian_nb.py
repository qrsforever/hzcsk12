#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file gaussian_nb.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-13 22:43

from k12ai.models.base import K12Algorithm
from sklearn.naive_bayes import GaussianNB as Algo


class SKGaussianNB(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
