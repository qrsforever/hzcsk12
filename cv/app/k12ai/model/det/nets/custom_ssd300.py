#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file custom_ssd300.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-04 16:20:44

from torch import nn

from model.det.nets.vgg16_ssd300 import (Vgg16SSD300, SSDHead, L2Norm)
from model.det.layers.ssd_detection_layer import SSDDetectionLayer
from model.det.layers.ssd_target_generator import SSDTargetGenerator

from k12ai.tools.util.net_def import load_custom_model
# from lib.tools.util.logger import Logger as Log


class CustomSSD300(Vgg16SSD300):
    def __init__(self, configer):
        super(Vgg16SSD300, self).__init__()
        self.configer = configer
        cache_dir = configer.get('network.checkpoints_root')
        model_name = self.configer.get('network.model_name')
        # self.backbone = load_custom_model(cache_dir, model_name).named_modules()
        self.backbone = [mod for mod in load_custom_model(cache_dir, model_name).children()]

        cnn_layers = [(i, mod) for i, mod in enumerate(self.backbone) if isinstance(mod, nn.Conv2d)]

        # check
        if len(cnn_layers) == 0 or cnn_layers[-1][1].out_channels != 1024:
            print('SSDModel1024')
            raise RuntimeError('SSDModel1024: the last cnn out channels must be 1024')

        seg_idx = -1
        for idx, mod in cnn_layers[::-1]:
            if mod.out_channels == 512:
                break
            seg_idx = idx
        if cnn_layers[0][0] == seg_idx:
            print('SSDModel512')
            raise RuntimeError('SSDModel512: must exist the cnn out channels 512')

        print(seg_idx)

        self.sub_backbone_1 = nn.ModuleList()
        self.sub_backbone_2 = nn.ModuleList()
        for i, module in enumerate(self.backbone):
            if i < seg_idx:
                self.sub_backbone_1.append(module)
            else:
                self.sub_backbone_2.append(module)

        self.norm4 = L2Norm(512, 20)
        self.ssd_head = SSDHead(configer)
        self.ssd_detection_layer = SSDDetectionLayer(configer)
        self.ssd_target_generator = SSDTargetGenerator(configer)
        self.valid_loss_dict = configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))
