#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file custom_ssd.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-05 16:47


from torch import nn

from model.det.nets.vgg16_ssd512 import (Vgg16SSD512, SSDHead, L2Norm)
from model.det.layers.ssd_detection_layer import SSDDetectionLayer
from model.det.layers.ssd_target_generator import SSDTargetGenerator

from k12cv.tools.util.net_def import load_custom_model


class CustomSSD512(Vgg16SSD512):
    def __init__(self, configer):
        super(Vgg16SSD512, self).__init__()
        self.configer = configer
        cache_dir = configer.get('network.checkpoints_root')
        model_name = self.configer.get('network.model_name')
        self.backbone = load_custom_model(cache_dir, model_name).named_modules()
        cnt = 0
        self.sub_backbone_1 = nn.ModuleList()
        self.sub_backbone_2 = nn.ModuleList()
        for key, module in self.backbone:
            if not key:
                continue
            if cnt < 23:
                self.sub_backbone_1.append(module)
            else:
                self.sub_backbone_2.append(module)

            cnt += 1

        self.norm4 = L2Norm(512, 20)
        self.ssd_head = SSDHead(configer)
        self.ssd_detection_layer = SSDDetectionLayer(configer)
        self.ssd_target_generator = SSDTargetGenerator(configer)
        self.valid_loss_dict = configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))
