#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file classic_agent.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-04-08 19:58


from rlpyt.agents.dqn.dqn_agent import DqnAgent
from k12ai.models.dqn.classic_dqn_model import ClassicDqnModel


class DiscreteMixin:

    def make_env_to_model_kwargs(self, env_spaces):
        d = dict(observation_shape=env_spaces.observation.shape,
                    action_size=env_spaces.action.n)
        return d


class ClassicDiscreteDqnAgent(DiscreteMixin, DqnAgent):

    def __init__(self, ModelCls=ClassicDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
