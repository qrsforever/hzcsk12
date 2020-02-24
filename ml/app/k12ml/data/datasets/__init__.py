#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file __init__.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-24 20:54

import os
from k12ml.utils.misc import find_components
from k12ml.data.base import K12DataLoader

_datasets = find_components(
    __package__,
    os.path.split(__file__)[0],
    K12DataLoader
)


def k12ai_get_loader(dataset):
    return _datasets[dataset]
