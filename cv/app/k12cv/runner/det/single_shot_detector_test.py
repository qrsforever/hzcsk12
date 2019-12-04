#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file single_shot_detector_test.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-04 10:56:27

import torch

from tools.util.logger import Logger as Log
from datasets.det.data_loader import DataLoader
from runner.tools.runner_helper import RunnerHelper
from model.det.model_manager import ModelManager
from metric.det.det_running_score import DetRunningScore
from tools.helper.dc_helper import DCHelper

from runner.det.single_shot_detector_test import SingleShotDetectorTest as SSDT
from k12cv.tools.util.rpc_message import hzcsk12_send_message

class SingleShotDetectorTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.det_model_manager = ModelManager(configer)
        self.det_data_loader = DataLoader(configer)
        self.det_running_score = DetRunningScore(configer)
        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = RunnerHelper.load_net(self, self.det_net)
        self.val_loader = self.det_data_loader.get_valloader('test')

    def train(self):
        Log.warn('no need')

    def test(self, test_dir, out_dir):
        self.det_net.eval()
        with torch.no_grad():
            for j, data_dict in enumerate(self.val_loader):
                data_dict = RunnerHelper.to_device(self, data_dict)
                out = self.det_net(data_dict)
                out_dict = RunnerHelper.gather(self, out)
                batch_detections = SSDT.decode(out_dict['loc'], out_dict['conf'],
                        self.configer, DCHelper.tolist(data_dict['meta']))
                batch_pred_bboxes = self.__get_object_list(batch_detections)
                self.det_running_score.update(batch_pred_bboxes,
                        [item['ori_bboxes'] for item in DCHelper.tolist(data_dict['meta'])],
                        [item['ori_labels'] for item in DCHelper.tolist(data_dict['meta'])])

            mAP = self.det_running_score.get_mAP()
            Log.info('Val mAP: {}'.format(mAP))
            hzcsk12_send_message('metrics', {'evaluate_mAP': mAP})

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
