#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file image_classifier_test.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 22:26:03

import time
import torch

from datasets.cls.data_loader import DataLoader
from runner.tools.runner_helper import RunnerHelper
from model.cls.model_manager import ModelManager
from tools.util.average_meter import AverageMeter, DictAverageMeter
from metric.cls.cls_running_score import ClsRunningScore
from tools.util.logger import Logger as Log

from k12cv.tools.util.rpc_message import hzcsk12_send_message

class ImageClassifierTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.runner_state = dict()

        self.batch_time = AverageMeter()
        self.cls_model_manager = ModelManager(configer)
        self.cls_data_loader = DataLoader(configer)
        self.running_score = ClsRunningScore(configer)

        self._init_model()

    def _init_model(self):
        self.cls_net = self.cls_model_manager.get_cls_model()
        self.cls_net = RunnerHelper.load_net(self, self.cls_net)
        # load 'test.json'
        self.val_loader = self.cls_data_loader.get_valloader('test')
        self.loss = self.cls_model_manager.get_cls_loss()

    def train(self):
        Log.warn("no need")

    def test(self, test_dir, out_dir):
        # don't need 'test_dir' and 'out_dir', only need test.json
        start_time = time.time()
        with torch.no_grad():
            for j, data_dict in enumerate(self.val_loader):
                data_dict = RunnerHelper.to_device(self, data_dict)
                out = self.cls_net(data_dict)
                out_dict, label_dict, loss_dict = RunnerHelper.gather(self, out)
                self.running_score.update(out_dict, label_dict)
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            top1 = RunnerHelper.dist_avg(self, self.running_score.get_top1_acc())
            top3 = RunnerHelper.dist_avg(self, self.running_score.get_top3_acc())
            top5 = RunnerHelper.dist_avg(self, self.running_score.get_top5_acc())
            if isinstance(top1, dict) and 'out' in top1.keys():
                top1 = top1['out']
                top3 = top3['out']
                top5 = top5['out']
            hzcsk12_send_message('metrics', {
                'evaluate_accuracy': top1,
                'evaluate_accuracy3': top3,
                'evaluate_accuracy5': top5,
                })
            Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))
            Log.info('Top1 ACC = {}'.format(top1))
            Log.info('Top3 ACC = {}'.format(top3))
            Log.info('Top5 ACC = {}'.format(top5))