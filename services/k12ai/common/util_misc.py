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
import signal
import torchvision # noqa
import numpy as np
from collections import OrderedDict


def install_signal_handler(signum, handler):
    old_signal_handler = None

    def _inner_handler(_signum, frame):
        handler(_signum, frame)
        signal.signal(signum, signal.SIG_DFL)
        if old_signal_handler not in (signal.SIG_IGN, signal.SIG_DFL):
            old_signal_handler(_signum, frame)

    old_signal_handler = signal.signal(signum, _inner_handler)


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


def sw_list(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def base64_image(image):
    if isinstance(image, str):
        path = image
    elif isinstance(image, (torch.Tensor, np.array)):
        path = '/tmp/__k12.png'
        torchvision.utils.save_image(image, path)
    else:
        raise NotImplementedError
    with open(path, 'rb') as fw:
        img = base64.b64encode(fw.read())
    return img.decode()


def make_histogram(values, bins=10):
    if isinstance(values, torch.autograd.Variable):
        values = values.data
    values = values.cpu().numpy().astype(float).reshape(-1)
    return np.histogram(values, bins=bins)
