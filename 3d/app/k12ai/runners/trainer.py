#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-21 13:16


from datetime import datetime
import shutil
import socket
import time
import torch
from torch.optim import lr_scheduler

import torch.nn as nn
import numpy as np


class Trainer(object):

    def train(self):
        raise NotImplementedError


class DepthPredictTrainer(Trainer):
    def __init__(
            self, 
            train_data_loader, val_data_loader,
            model, optimizer, scheduler,
            num_epochs, cache_dir):
        pass
