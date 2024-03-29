#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file __init__.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-14 23:13

import os
from k12ai.common.util_misc import find_components
from k12ai.models.base import K12Algorithm

_clusters = find_components(
    __package__,
    os.path.split(__file__)[0],
    K12Algorithm
)


def k12ai_get_model(model_name):
    return _clusters[model_name]
