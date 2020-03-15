#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file util_misc.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-04 17:28

import importlib
import inspect
import pkgutil
import sys
import base64
import torch
import torchvision # noqa
import numpy as np
from collections import OrderedDict


def find_components(package, directory, base_class):
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    components[module_name] = obj 
    return components


def base64_image(path):
    with open(path,'rb') as fw:
        img = base64.b64encode(fw.read())
    return img.decode()


def make_histogram(values, bins=10):
    if isinstance(values, torch.autograd.Variable):
        values = values.data
    values = values.cpu().numpy().astype(float).reshape(-1)
    return np.histogram(values, bins=bins)
