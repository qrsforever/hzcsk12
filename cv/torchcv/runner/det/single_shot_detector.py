#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Single Shot Detector.


import time
import torch

from data.det.data_loader import DataLoader
from runner.det.single_shot_detector_test import SingleShotDetectorTest
from lib.runner.runner_helper import RunnerHelper
from lib.runner.trainer import Trainer
from model.det.model_manager import ModelManager
from lib.tools.util.average_meter import AverageMeter
from lib.tools.util.logger import Logger as Log
from metric.det.det_running_score import DetRunningScore
from lib.tools.vis.det_visualizer import DetVisualizer
from lib.tools.helper.dc_helper import DCHelper

# QRS: add
from k12ai.runner.stat import RunnerStat


class SingleShotDetector(object):
    """
      The class for Single Shot Detector. Include train, val, test & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.det_visualizer = DetVisualizer(configer)
        self.det_model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.det_running_score = DetRunningScore(configer)

        self.det_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.runner_state = dict()

        self._init_model()

    def _init_model(self):
        # torch.multiprocessing.set_sharing_strategy('file_system')
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = RunnerHelper.load_net(self, self.det_net)
        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.configer.get('solver'))
        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()
        self.det_loss = self.det_model_manager.get_det_loss()

    def _get_parameters(self):
        lr_1 = []
        lr_10 = []
        params_dict = dict(self.det_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                lr_10.append(value)
            else:
                lr_1.append(value)

        params = [{'params': lr_1, 'lr': self.configer.get('solver', 'lr')['base_lr']},
                  {'params': lr_10, 'lr': self.configer.get('solver', 'lr')['base_lr'] * 1.0}]

        return params

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.det_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.runner_state['epoch'] += 1

        # data_tuple: (inputs, heatmap, maskmap, vecmap)
        for i, data_dict in enumerate(self.train_loader):
            self.data_time.update(time.time() - start_time)
            # Forward pass.
            data_dict = RunnerHelper.to_device(self, data_dict)
            out = self.det_net(data_dict)
            loss_dict = self.det_loss(out)
            loss = loss_dict['loss']
            self.train_losses.update(loss.item(), len(DCHelper.tolist(data_dict['meta'])))

            self.optimizer.zero_grad()
            loss.backward()
            RunnerHelper.clip_grad(self.det_net, 10.)
            self.optimizer.step()

            # QRS
            Trainer.update(self, warm_list=(0,),
                           warm_lr_list=(self.configer.get('solver', 'lr')['base_lr'],),
                           solver_dict=self.configer.get('solver'))

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.runner_state['iters'] += 1

            # Print the log info & reset the states.
            if self.runner_state['iters'] % self.configer.get('solver', 'display_iter') == 0:
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    self.runner_state['epoch'], self.runner_state['iters'],
                    self.configer.get('solver', 'display_iter'),
                    RunnerHelper.get_lr(self.optimizer), batch_time=self.batch_time,
                    data_time=self.data_time, loss=self.train_losses))

                # QRS: add
                RunnerStat.train(self, data_dict)

                self.batch_time.reset()
                self.data_time.reset()

            if self.configer.get('solver', 'lr')['metric'] == 'iters' \
                    and self.runner_state['iters'] == self.configer.get('solver', 'max_iters'):
                break

            # Check to val the current model.
            if self.runner_state['iters'] % self.configer.get('solver', 'test_interval') == 0:
                self.val()
                self.train_losses.reset()

    def val(self):
        """
          Validation function during the train phase.
        """
        self.det_net.eval()
        start_time = time.time()
        with torch.no_grad():
            for j, data_dict in enumerate(self.val_loader):
                # Forward pass.
                data_dict = RunnerHelper.to_device(self, data_dict)
                out = self.det_net(data_dict)
                loss_dict = self.det_loss(out)
                loss = loss_dict['loss']
                out_dict, _ = RunnerHelper.gather(self, out)
                # Compute the loss of the val batch.
                self.val_losses.update(loss.item(), len(DCHelper.tolist(data_dict['meta'])))

                batch_detections = SingleShotDetectorTest.decode(out_dict['loc'], out_dict['conf'],
                                                                 self.configer, DCHelper.tolist(data_dict['meta']))
                batch_pred_bboxes = self.__get_object_list(batch_detections)
                # batch_pred_bboxes = self._get_gt_object_list(batch_gt_bboxes, batch_gt_labels)
                self.det_running_score.update(batch_pred_bboxes,
                                              [item['ori_bboxes'] for item in DCHelper.tolist(data_dict['meta'])],
                                              [item['ori_labels'] for item in DCHelper.tolist(data_dict['meta'])])

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            # QRS:
            RunnerStat.validation(self)
            RunnerHelper.save_net(self, self.det_net, iters=self.runner_state['iters'])
            # Print the log info & reset the states.
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            Log.info('Val mAP: {}'.format(self.det_running_score.get_mAP()))
            self.det_running_score.reset()
            self.batch_time.reset()
            self.val_losses.reset()
            self.det_net.train()

    def __get_object_list(self, batch_detections):
        batch_pred_bboxes = list()
        for idx, detections in enumerate(batch_detections):
            object_list = list()
            if detections is not None:
                for x1, y1, x2, y2, conf, cls_pred in detections:
                    cf = float('%.2f' % conf.item())
                    cls_pred = int(cls_pred.cpu().item() - 1)
                    object_list.append([x1.item(), y1.item(), x2.item(), y2.item(), cls_pred, cf])

            batch_pred_bboxes.append(object_list)

        return batch_pred_bboxes


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
