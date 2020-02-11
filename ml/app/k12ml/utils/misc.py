#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file misc.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 20:01

import importlib
import inspect
import pkgutil
import sys
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
