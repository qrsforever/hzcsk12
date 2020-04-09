#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file classic_dqn_model.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-04-08 20:52

import torch
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


# TODO
class ClassicDqnModel(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=64,
            **kwargs):
        super().__init__()
        self._obs_ndim = len(image_shape)
        input_shape = image_shape[0]

        self.base_net = torch.nn.Sequential(
            torch.nn.Linear(input_shape, fc_sizes),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_sizes, fc_sizes),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_sizes, output_size),
        )

    def forward(self, observation, prev_action, prev_reward):
        observation = observation.type(torch.float)
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, self._obs_ndim)
        obs = observation.view(T * B, -1)
        q = self.base_net(obs)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
