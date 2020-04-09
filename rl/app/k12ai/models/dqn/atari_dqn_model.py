#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file atari_dqn_model.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-04-09 20:35
# @see rlpyt/rlpyt/models/dqn/atari_dqn_model.py

import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


class AtariDqnModel(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,
            dueling=False,
            use_maxpool=False,
            channels=None,
            kernel_sizes=None,
            strides=None,
            paddings=None):
        super().__init__()
        self.dueling = dueling
        h, w, c = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings or [0, 1, 1],
            use_maxpool=use_maxpool,
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        if dueling:
            self.head = DuelingHeadModel(conv_out_size, fc_sizes, output_size)
        else:
            self.head = MlpModel(conv_out_size, fc_sizes, output_size)

    def forward(self, observation, prev_action, prev_reward):
        img = observation.type(torch.float)
        img = img.mul_(1. / 255)
        
        # QRS
        if len(observation.shape) == 4:
            img = img.permute(0, 3, 1, 2)
        else:
            img = img.permute(2, 0, 1)

        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))
        q = self.head(conv_out.reshape(T * B, -1))
        q = restore_leading_dims(q, lead_dim, T, B)
        return q
