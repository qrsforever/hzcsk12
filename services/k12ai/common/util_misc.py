#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file util_misc.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-04 17:28

import os
import importlib
import inspect
import traceback # noqa
import pkgutil
import sys
import io
import torch
import base64
import math
import numbers
import random
import signal
import torchvision # noqa
import numpy as np
import PIL
from itertools import cycle
from collections import OrderedDict
from torchvision import transforms

import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D # noqa
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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


def handle_exception(handler):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                if handler:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    handler(exc_type.__name__, str(exc_value), exc_tb)
                return None
        return wrapper
    return decorator


## Tools

def sw_list(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def set_rng_seed(rng_seed):
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    try:
        np.random.seed(rng_seed)
    except ImportError:
        pass


def torch_isnan(x):
    if isinstance(x, numbers.Number):
        return x != x
    return torch.isnan(x).any()


def torch_isinf(x):
    if isinstance(x, numbers.Number):
        return x == math.inf or x == -math.inf
    return (x == math.inf).any() or (x == -math.inf).any()


def time_to_DHMS(time_sec):
    days, hours, minutes, seconds = 0, 0, 0, 0
    if time_sec > 86400:
        days = int(time_sec // 86400)
        time_sec = time_sec % 86400
    if time_sec > 3600:
        hours = int(time_sec // 3600)
        time_sec = time_sec % 3600
    if time_sec > 60:
        minutes = int(time_sec // 60)
        seconds = time_sec % 60
    return days, hours, minutes, seconds


def transform_denormalize(inputs, mean, std, inplace=False, div_value=1.0):
    demean = [-m / s for m, s in zip(mean, std)]
    destd = [1 / s for s in std]
    inputs = transforms.Normalize(demean, destd, inplace)(inputs)
    return torch.clamp(inputs, 0.0, 1.0)


def image2bytes(image, width=None, height=None):
    if isinstance(image, bytes):
        return image

    if isinstance(image, Figure):
        with io.BytesIO() as fw:
            plt.savefig(fw)
            return fw.getvalue()

    if isinstance(image, str):
        if os.path.isfile(image):
            image = PIL.Image.open(image).convert("RGB")
        else:
            # TODO: SVG
            import cairosvg
            with io.BytesIO() as fw:
                cairosvg.svg2png(bytestring=image, write_to=fw,
                        output_width=width, output_height=height)
                return fw.getvalue()

    elif isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    elif isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image.astype('uint8')).convert('RGB')

    if isinstance(image, PIL.Image.Image):
        if width and height:
            image = image.resize((width, height))
        bio = io.BytesIO()
        image.save(bio, "PNG")
        bio.seek(0)
        return bio.read()

    raise NotImplementedError(type(image))


def img2b64(x):
    return base64.b64encode(image2bytes(x)).decode()


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


## Draw

def plot_decision_boundaries(xx, yy, zz, X0, X1, Y, C0=None, C1=None):
    plt.clf()
    fig, ax = plt.subplots(dpi=150)
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


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.clf()
    fig = plt.figure(dpi=150)
    colors = cycle(['red', 'green', 'blue', 'aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(roc_auc)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC to multi-class')
    plt.legend(loc="lower right")
    return fig


def plot_regression3D(X, Y, Z0, Z1, zlabel=None):
    plt.clf()
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z0, marker='^', label='true')
    ax.scatter(X, Y, Z1, marker='o', label='pred')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if zlabel:
        ax.set_zlabel(zlabel)
    else:
        ax.set_zlabel('Z')
    plt.legend()
    return fig


def dr_scatter2D(data, labels):
    if data.shape[1] > 50:
        pca_50 = PCA(n_components=50)
        data = pca_50.fit_transform(data)

    tsne = TSNE(n_components=2, random_state=21).fit_transform(data)
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111)

    colors = cm.rainbow(np.linspace(0, 1, 1))

    for i, (x, y) in enumerate(tsne):
        ax.scatter(x, y, lw=0, s=40, alpha=0.5, c=colors, edgecolor='none')
        ax.text(x, y, labels[i], fontsize=2)
    return fig


def dr_scatter3D(data, labels):
    if data.shape[1] > 50:
        pca_50 = PCA(n_components=50)
        data = pca_50.fit_transform(data)

    tsne = TSNE(n_components=3, random_state=21).fit_transform(data)
    fig = plt.figure(dpi=150)
    ax = Axes3D(fig)

    colors = cm.rainbow(np.linspace(0, 1, 1))
    ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2], c=colors, alpha=0.5)
    return fig


def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)
