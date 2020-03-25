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
import io
import base64
import torch
import signal
import torchvision # noqa
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


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
        with open(image, 'rb') as fw:
            rawbytes = fw.read()
    elif isinstance(image, Figure):
        with io.BytesIO() as fw:
            plt.savefig(fw)
            rawbytes = fw.getvalue()
    elif isinstance(image, (torch.Tensor, np.array)):
        with io.BytesIO() as fw:
            torchvision.utils.save_image(image, fw, format='png')
            rawbytes = fw.getvalue()
    else:
        raise NotImplementedError
    return base64.b64encode(rawbytes).decode()


def make_histogram(values, bins=10):
    if isinstance(values, torch.autograd.Variable):
        values = values.data
    values = values.cpu().numpy().astype(float).reshape(-1)
    return np.histogram(values, bins=bins)


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_decision_boundaries(xx, yy, zz, X0, X1, Y, C0=None, C1=None):
    plt.clf()
    fig, ax = plt.subplots(dpi=100)
    ax.contourf(xx, yy, zz.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    if C0 is not None and C1 is not None:
        ax.scatter(C0, C1, c='w', marker='x', s=169, linewidths=3, zorder=10)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_xticks(())
    ax.set_yticks(())
    return fig
