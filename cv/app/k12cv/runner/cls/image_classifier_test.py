#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file image_classifier_test.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 22:26:03

import os
import cv2
import json
import torch
import torchvision.transforms as transforms
from PIL import Image

from datasets.test.test_data_loader import TestDataLoader
from runner.tools.blob_helper import BlobHelper
from runner.tools.runner_helper import RunnerHelper
from model.cls.model_manager import ModelManager
from tools.helper.image_helper import ImageHelper
from tools.helper.json_helper import JsonHelper
from tools.util.logger import Logger as Log
from tools.parser.cls_parser import ClsParser

class ImageClassifierTest(object):

    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.cls_model_manager = ModelManager(configer)
        self.test_loader = TestDataLoader(configer)
        self.cls_parser = ClsParser(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.cls_net = None
        if self.configer.get('dataset') == 'imagenet':
            with open(os.path.join(self.configer.get('project_dir'),
                                   'datasets/cls/imagenet/imagenet_class_index.json')) as json_stream:
                name_dict = json.load(json_stream)
                name_seq = [name_dict[str(i)][1] for i in range(self.configer.get('data', 'num_classes'))]
                self.configer.add(['details', 'name_seq'], name_seq)

        self._init_model()

    def _init_model(self):
        self.cls_net = self.cls_model_manager.get_cls_model()
        self.cls_net = RunnerHelper.load_net(self, self.cls_net)
        self.cls_net.eval()

    def test(self, test_dir, out_dir):
        for _, data_dict in enumerate(self.test_loader.get_testloader(test_dir=test_dir)):
           Logger.info(data_dict) 
