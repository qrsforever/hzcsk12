#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

import importlib
import os


def load_model(proj_dir, model_name):
    """load custom model according to net name
  
  Args:
    task (string): task of particular model 
    net_name (string): name of custom neural net
  
  Returns:
    nn.module: an instanc of custom model
  """

    py_file = "{0}/custom/{1}.py".format(proj_dir, model_name)
    spec = importlib.util.spec_from_file_location(model_name, py_file)
    custom_net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_net)
    custom_model = custom_net.custom_model()
    return custom_model


def gen_model_urls(weights_host=None, model_files=None):
    assert weights_host is not None, "weights host must be specified"
    assert (
        len(model_files) > 0
    ), "at least one pretrained model shoud be provided"
    model_urls = {}
    for key, value in model_files.items():
        model_urls[key] = weights_host + value
    return model_urls


def load_genotype(file_path, epoch):
    """load genotype from disks
  
  Args:
    file_path: path to file 
    epoch: num of specific epoch
  
  Returns :
    genotype: genotype describes an cell 
  """
    print(file_path)
    spec = importlib.util.spec_from_file_location("genotypes", file_path)
    genotypes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(genotypes)
    return eval("genotypes.DARTS_EPOCH_{}".format(epoch))
