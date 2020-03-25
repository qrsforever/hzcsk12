#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file svc.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 18:33

from k12ai.models.base import K12Algorithm
from sklearn.svm import SVC as Algo


class SKSVC(K12Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(Algo, *args, **kwargs)
