#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file scheduler.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-24 22:20

import torch

SCHEDULER_DICT = {
    'reduceonplateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'step': torch.optim.lr_scheduler.StepLR,
    'multistep': torch.optim.lr_scheduler.MultiStepLR,
}
