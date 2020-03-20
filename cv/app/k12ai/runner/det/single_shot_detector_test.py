#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file single_shot_detector_test.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-04 10:56:27

import torch
from torch.utils import data

from lib.tools.util.logger import Logger as Log
# from data.det.data_loader import DataLoader
from lib.runner.runner_helper import RunnerHelper
from model.det.model_manager import ModelManager
from metric.det.det_running_score import DetRunningScore
from lib.tools.helper.dc_helper import DCHelper
# from data.test.test_data_loader import TestDataLoader
from data.det.datasets.default_dataset import DefaultDataset
import lib.data.pil_aug_transforms as pil_aug_trans
import lib.data.cv2_aug_transforms as cv2_aug_trans
import lib.data.transforms as trans
from lib.data.collate import collate

from runner.det.single_shot_detector_test import SingleShotDetectorTest as SSDT
from k12ai.runner.stat import RunnerStat


class SingleShotDetectorTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.det_model_manager = ModelManager(configer)
        # self.det_data_loader = DataLoader(configer)
        # self.test_loader = TestDataLoader(configer)
        self.det_running_score = DetRunningScore(configer)

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = RunnerHelper.load_net(self, self.det_net)
        # self.val_loader = self.det_data_loader.('test')

    def train(self):
        Log.warn('no need')

    def test(self, test_dir, out_dir):
        # self.configer.add('test.dataset', 'json')
        # self.configer.add('test.json_path', '%s/test.json' % self.configer.get('data.data_dir'))
        # self.configer.add('test.root_dir', self.configer.get('data.data_dir'))
        self.det_net.eval()
        with torch.no_grad():
            for j, data_dict in enumerate(self.__get_valloader()):
                data_dict = RunnerHelper.to_device(self, data_dict)
                out = self.det_net(data_dict)
                out_dict = RunnerHelper.gather(self, out)
                batch_detections = SSDT.decode(out_dict['loc'], out_dict['conf'],
                        self.configer, DCHelper.tolist(data_dict['meta']))
                batch_pred_bboxes = self.__get_object_list(batch_detections)
                self.det_running_score.update(batch_pred_bboxes,
                        [item['ori_bboxes'] for item in DCHelper.tolist(data_dict['meta'])],
                        [item['ori_labels'] for item in DCHelper.tolist(data_dict['meta'])])

            RunnerStat.evaluate(self)
            mAP = self.det_running_score.get_mAP()
            Log.info('Test mAP: {}'.format(mAP))

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

    def __get_valloader(self):
        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_test_transform = pil_aug_trans.PILAugCompose(self.configer, split='test')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_test_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='test')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)
        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(**self.configer.get('data', 'normalize')), ])

        if self.configer.get('dataset', default=None) in [None, 'default']:
            dataset = DefaultDataset(root_dir=self.configer.get('data', 'data_dir'), dataset='test',
                                     aug_transform=self.aug_test_transform,
                                     img_transform=self.img_transform,
                                     configer=self.configer)

        else:
            Log.error('{} dataset is invalid.'.format(self.configer.get('dataset')))
            exit(1)

        testloader = data.DataLoader(
            dataset,
            batch_size=self.configer.get('test', 'batch_size'), shuffle=False,
            num_workers=self.configer.get('data', 'workers'), pin_memory=True,
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('test', 'data_transformer')
            )
        )
        return testloader
