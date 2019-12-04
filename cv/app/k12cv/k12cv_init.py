#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12cv_init.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 19:38:04

import os

from tools.util.configer import Configer
from tools.util.logger import Logger as Log

from k12cv.runner.cls.image_classifier_test import ImageClassifierTest
from k12cv.runner.det.single_shot_detector_test import SingleShotDetectorTest
from k12cv.model.det.nets.custom_ssd300 import CustomSSD300

from k12cv.tools.util.net_def import build_custom_model

from runner.runner_selector import CLS_TEST_DICT, DET_TEST_DICT
from model.det.model_manager import DET_MODEL_DICT 

# change the original method to hzcsk12 hook
def _hook_runner_selector(configer):
    task = configer.get('task')
    Log.info("_hook_runner_selector(%s)" % task)
    if task == 'cls':
        CLS_TEST_DICT['image_classifier'] = ImageClassifierTest
    elif task == 'det':
        DET_TEST_DICT['single_shot_detector'] = SingleShotDetectorTest
        DET_MODEL_DICT['custom_ssd300'] = CustomSSD300
    else:
        Log.info("not impl")

# build custom model if model name begin with the prefix 'custom_'
def _check_custom_model(configer):
    model_name = configer.get('network.model_name')
    Log.info('model name[%s]' % model_name)
    if model_name.split("_")[0] == "custom":
        cache_dir = configer.get('cache_dir')
        net_def_str = configer.get('network.net_def')
        net_def_dir = build_custom_model(cache_dir, net_def_str, model_name)
        net_def_fil = os.path.join(net_def_dir, '%s.txt' % model_name)
        with open(net_def_fil, 'w') as fout:
            fout.write(net_def_str)

def hzcsk12_cv_init(configer):

    Log.info('hzcsk12_cv_init')

    _check_custom_model(configer)

    _hook_runner_selector(configer)
