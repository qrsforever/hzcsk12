#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file image_classifier_test.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 22:26:03

import torch

from data.cls.data_loader import DataLoader
from lib.runner.runner_helper import RunnerHelper
from model.cls.model_manager import ModelManager
from metric.cls.cls_running_score import ClsRunningScore
from lib.tools.util.logger import Logger as Log

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from k12ai.common.log_message import MessageReport
from k12ai.runner.stat import RunnerStat


class ImageClassifierTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.runner_state = dict()

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
        Log.warn('no need')

    def test(self, test_dir, out_dir):
        # don't need 'test_dir' and 'out_dir', only need test.json
        with torch.no_grad():
            targets_list = []
            predicted_list = []
            path_list = []
            for j, data_dict in enumerate(self.val_loader):
                data_dict = RunnerHelper.to_device(self, data_dict)
                out = self.cls_net(data_dict)
                out_dict, label_dict, loss_dict = RunnerHelper.gather(self, out)
                self.running_score.update(out_dict, label_dict)
                targets_list.append(label_dict['out'].cpu())
                predicted_list.append(out_dict['out'].max(1)[1].cpu())
                path_list.append(data_dict['path'])

            targets, predicted = torch.cat(targets_list), torch.cat(predicted_list)
            print(confusion_matrix(targets, predicted))
            print(precision_recall_fscore_support(targets, predicted, average='macro'))

            RunnerStat.evaluate(self, targets, predicted, path_list)

            top1 = RunnerHelper.dist_avg(self, self.running_score.get_top1_acc())
            top3 = RunnerHelper.dist_avg(self, self.running_score.get_top3_acc())
            top5 = RunnerHelper.dist_avg(self, self.running_score.get_top5_acc())
            if isinstance(top1, dict) and 'out' in top1.keys():
                top1 = top1['out']
                top3 = top3['out']
                top5 = top5['out']
            MessageReport.metrics({
                'evaluate_iters': 1,
                'evaluate_progress': 1.0,
                'evaluate_accuracy': top1,
                'evaluate_accuracy3': top3,
                'evaluate_accuracy5': top5})
            Log.info('Top1 ACC = {}'.format(top1))
            Log.info('Top3 ACC = {}'.format(top3))
            Log.info('Top5 ACC = {}'.format(top5))
