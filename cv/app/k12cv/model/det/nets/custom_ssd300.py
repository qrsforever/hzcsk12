#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file custom_ssd300.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-04 16:20:44

import torch
from torch import nn
import torch.nn.init as init

from model.det.layers.ssd_detection_layer import SSDDetectionLayer
from model.det.layers.ssd_target_generator import SSDTargetGenerator
from model.det.loss.loss import BASE_LOSS_DICT
from tools.util.logger import Logger as Log

from k12cv.tools.util.net_def import load_custom_model

DETECTOR_CONFIG = {
    "num_centrals": [256, 128, 128, 128],
    "num_strides": [2, 2, 1, 1],
    "num_padding": [1, 1, 0, 0],
}

class CustomSSD300(nn.Module):
    def __init__(self, configer):
        super(CustomSSD300, self).__init__()
        self.configer = configer
        cache_dir = configer.get('cache_dir')
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

    def forward(self, data_dict):
        x = data_dict['img']
        out = []
        for module in self.sub_backbone_1:
            x = module(x)

        out.append(self.norm4(x))
        for module in self.sub_backbone_2:
            x = module(x)

        out.append(x)
        out_head = self.ssd_head(x)
        final_out = out + out_head

        pred_loc, pred_conf, dets_loc, dets_conf = self.ssd_detection_layer(final_out, data_dict)
        out_dict = dict(loc=dets_loc, conf=dets_conf)
        if self.configer.get('phase') == 'test':
            return out_dict

        loc_targets, conf_targets = self.ssd_target_generator(final_out, data_dict)
        loss_dict = dict()
        if 'multibox_loss' in self.valid_loss_dict:
            loss_dict['multibox_loss'] = dict(
                params=[pred_loc, pred_conf, loc_targets, conf_targets],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['multibox_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['multibox_loss']])
            )
        return out_dict, loss_dict

class SSDHead(nn.Module):

    def __init__(self, configer):
        super(SSDHead, self).__init__()

        self.configer = configer
        self.num_features = self.configer.get('network', 'num_feature_list')
        self.num_centrals = DETECTOR_CONFIG['num_centrals']
        self.num_paddings = DETECTOR_CONFIG['num_padding']
        self.num_strides = DETECTOR_CONFIG['num_strides']

        self.feature2 = self.__extra_layer(num_in=self.num_features[1], num_out=self.num_features[2],
                num_c=self.num_centrals[0], stride=self.num_strides[0],
                pad=self.num_paddings[0])
        self.feature3 = self.__extra_layer(num_in=self.num_features[2], num_out=self.num_features[3],
                num_c=self.num_centrals[1], stride=self.num_strides[1],
                pad=self.num_paddings[1])
        self.feature4 = self.__extra_layer(num_in=self.num_features[3], num_out=self.num_features[4],
                num_c=self.num_centrals[2], stride=self.num_strides[2],
                pad=self.num_paddings[2])
        self.feature5 = self.__extra_layer(num_in=self.num_features[4], num_out=self.num_features[5],
                num_c=self.num_centrals[3], stride=self.num_strides[3],
                pad=self.num_paddings[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def __extra_layer(num_in, num_out, num_c, stride, pad):
        layer = nn.Sequential(
                nn.Conv2d(num_in, num_c, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(num_c, num_out, kernel_size=3, stride=stride, padding=pad),
                nn.ReLU(),
                )
        return layer

    def forward(self, feature):
        det_feature = list()

        feature = self.feature2(feature)
        det_feature.append(feature)

        feature = self.feature3(feature)
        det_feature.append(feature)

        feature = self.feature4(feature)
        det_feature.append(feature)

        feature = self.feature5(feature)
        det_feature.append(feature)

        return det_feature

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
