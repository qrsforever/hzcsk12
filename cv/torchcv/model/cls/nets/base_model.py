#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch.nn as nn

from lib.model.module_helper import ModuleHelper
# from lib.tools.util.logger import Logger as Log


class BaseModel(nn.Module):
    def __init__(self, configer, flag=''):
        super(BaseModel, self).__init__()
        self.configer = configer
        # QRS: fix.
        num_classes = self.configer.get('data.num_classes')
        backbone = configer.get('network.backbone')
        pretrained_file = configer.get('network.pretrained')
        self.net = ModuleHelper.get_backbone(backbone=backbone, pretrained=pretrained_file)
        if backbone.startswith('vgg'):
            self.net.classifier[6] = nn.Linear(4096, num_classes)
        elif backbone.startswith('resnet'):
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        elif backbone.startswith('alexnet'):
            self.net.classifier[6] = nn.Linear(4096, num_classes)
        elif backbone.startswith('googlenet'):
            # QRS: not ok
            raise NotImplementedError(f'backbone:{backbone}')
            self.net.aux1.fc2 = nn.Linear(1024, num_classes)
            self.net.aux2.fc2 = nn.Linear(1024, num_classes)
            self.net.fc = nn.Linear(1024, num_classes)
        elif backbone.startswith('mobilenet_v2'):
            self.net.classifier[1] = nn.Linear(self.net.classifier[1].in_features, num_classes)
        elif backbone.startswith('squeezenet'):
            self.net.num_classes = num_classes
            self.net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        elif backbone.startswith('shufflenet'):
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        elif backbone.startswith('densenet'):
            in_features = {
                "densenet121": 1024,
                "densenet161": 2208,
                "densenet169": 1664,
                "densenet201": 1920,
            }
            self.net.classifier = nn.Linear(in_features[backbone], num_classes)
        else:
            raise NotImplementedError(f'backbone:{backbone}')

        self.valid_loss_dict = configer.get('loss.loss_weights', configer.get('loss.loss_type'))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.net.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.net.load_state_dict(state_dict, strict)

    # QRS: mod
    def forward(self, inputs):
        return self.net(inputs)
