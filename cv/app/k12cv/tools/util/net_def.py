#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file custom_model.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-04 19:38:23

import importlib
import os

from vulkan.builders.net_builder import NetBuilder

def build_custom_model(cache_dir, net_def, model_name):
    save_dir = '{}/custom'.format(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net_builder = NetBuilder(net_def)
    net = net_builder.build_net()
    net.write_net('{}/{}.py'.format(save_dir, model_name))
    return save_dir

def load_custom_model(cache_dir, model_name):
    save_dir = '{}/custom'.format(cache_dir)
    py_file = '{}/{}.py'.format(save_dir, model_name)
    spec = importlib.util.spec_from_file_location(model_name, py_file)
    custom_net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_net)
    custom_model = custom_net.custom_model()
    return custom_model
