#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file discrete_wrapper.py
# @brief
# @author QRS
# @blog qrsforever.github.io
# @version 1.0
# @date 2020-02-20 23:34

import numpy as np
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper


class DiscreteSpaceWrapper(GymSpaceWrapper):
    def revert(self, value):
        newval = abs(np.round(value[0])) if len(value.shape) > 0 else np.round(value)
        if newval >= self.n:
            newval = self.n - 1
        return newval

    @property
    def shape(self):
        return (1,)

    @property
    def low(self):
        return -1

    @property
    def high(self):
        return 1
