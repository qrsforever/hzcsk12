#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12cv_init.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 19:38:04

from tools.util.logger import Logger as Log

from k12cv.runner.cls.image_classifier_test import ImageClassifierTest

from runner.runner_selector import CLS_TEST_DICT

def hzcsk12_cv_init():
    # Change the image_classifier test hander
    CLS_TEST_DICT['image_classifier'] = ImageClassifierTest

