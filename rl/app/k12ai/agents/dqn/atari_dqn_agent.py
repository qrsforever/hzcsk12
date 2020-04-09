#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file atari_dqn_agent.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-04-09 20:41
# @see rlpyt/rlpyt/agents/atari_dqn_agent.py


from rlpyt.agents.dqn.dqn_agent import DqnAgent
from k12ai.models.dqn.atari_dqn_model import AtariDqnModel
from rlpyt.agents.dqn.atari.mixin import AtariMixin


class AtariDqnAgent(AtariMixin, DqnAgent):

    def __init__(self, ModelCls=AtariDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
