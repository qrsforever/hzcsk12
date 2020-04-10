#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn as nn

from lib.model.module_helper import ModuleHelper
from model.cls.loss.loss import BASE_LOSS_DICT


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
        else:
            raise NotImplementedError(f'backbne:{backbone}')

        self.valid_loss_dict = configer.get('loss.loss_weights', configer.get('loss.loss_type'))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.net.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.net.load_state_dict(state_dict, strict)

    def forward(self, data_dict):
        out_dict = dict()
        label_dict = dict()
        loss_dict = dict()
        # QRS: fix error
        # in_img = ModuleHelper.preprocess(data_dict['img'], self.configer.get('data.normalize'))
        # out = self.net(in_img)
        # out_dict['out'] = out
        # label_dict['out'] = data_dict['label'][:, 0]
        out = self.net(data_dict['img'])
        out_dict['out'] = out
        label_dict['out'] = data_dict['label']
        if 'ce_loss' in self.valid_loss_dict:
            loss_dict['ce_loss'] = dict(
                # params=[out, data_dict['label'][:, 0]],
                params=[out, data_dict['label']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['ce_loss']])
            )

        return out_dict, label_dict, loss_dict
