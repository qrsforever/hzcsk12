#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file classic_agent.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-04-08 19:58


from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.dqn.atari.mixin import AtariMixin
from k12ai.models.dqn.classic_dqn_model import ClassicDqnModel


class ClassicDiscreteDqnAgent(AtariMixin, DqnAgent):

    def __init__(self, ModelCls=ClassicDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
