#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 20:59


class BaseRunner:
    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
