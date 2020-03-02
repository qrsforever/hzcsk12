#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file custom_base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-06 18:30

from model.cls.nets.base_model import BaseModel

from k12ai.tools.util.net_def import load_custom_model


class CustomBaseModel(BaseModel):
    def __init__(self, configer):
        super(BaseModel, self).__init__()
        self.configer = configer
        cache_dir = configer.get('network.checkpoints_root')
        model_name = self.configer.get('network.model_name')
        self.net = load_custom_model(cache_dir, model_name)
        self.valid_loss_dict = configer.get('loss.loss_weights', configer.get('loss.loss_type'))
