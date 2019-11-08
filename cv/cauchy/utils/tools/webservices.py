#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import socket

from vulkan.builders.net_builder import NetBuilder


def gen_custom_model(net_def, save_dir):
    """generate custom model according to user design
  
  Args:
    net_def (string): description about user design of custom network
    save_dir (string): target destination to save network
  """

    net_builder = NetBuilder(net_def)
    net = net_builder.build_net()
    if not os.path.exists("{0}/custom".format(save_dir)):
        os.makedirs("{0}/custom".format(save_dir))
    net.write_net("{}/custom/{}.py".format(save_dir, net.proto.name))


def check_viz_status(host="127.0.0.1", port=8097):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        return True
    except Exception as e:
        print(e)
        return False
