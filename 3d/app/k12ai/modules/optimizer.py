#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file optimizer.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-24 22:20

import torch

OPTIMIZER_DICT = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
}
