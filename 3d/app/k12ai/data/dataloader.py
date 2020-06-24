#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file dataloader.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-24 19:55


class Dataloader(object):
    def get_trainloader(self):
        raise RuntimeError('subclass not impl.')

    def get_validloader(self):
        raise RuntimeError('subclass not impl.')

    def get_testloader(self):
        raise RuntimeError('subclass not impl.')
