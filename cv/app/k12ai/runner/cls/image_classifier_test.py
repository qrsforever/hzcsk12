#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file image_classifier_test.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 22:26:03

import os
import torch
import torch.nn.functional as F
from torch.utils import data

from data.cls.datasets.default_dataset import DefaultDataset
from lib.data.collate import collate
from lib.runner.runner_helper import RunnerHelper
from model.cls.model_manager import ModelManager
from metric.cls.cls_running_score import ClsRunningScore
from lib.tools.util.logger import Logger as Log
from lib.tools.helper.image_helper import ImageHelper
from lib.parallel.data_container import DataContainer

import lib.data.pil_aug_transforms as pil_aug_trans
import lib.data.cv2_aug_transforms as cv2_aug_trans
from torchvision import transforms as trans

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from k12ai.runner.stat import RunnerStat


class DirListDataset(data.Dataset):
    def __init__(self, root_dir=None, dataset=None, aug_transform=None, img_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.tool = configer.get('data', 'image_tool')
        self.mode = configer.get('data', 'input_mode')
        self.img_list = self.__list_dirs(root_dir)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index], tool=self.tool, mode=self.mode)

        if self.aug_transform is not None:
            img = self.aug_transform(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(
            img=DataContainer(img, stack=True),
            path=DataContainer(self.img_list[index], stack=True)
        )

    def __len__(self):
        return len(self.img_list)

    def __list_dirs(self, image_dir):
        img_list = list()
        for file_name in os.listdir(image_dir):
            img_list.append(f'{image_dir}/{file_name}')
        print(img_list)
        return img_list


class ImageClassifierTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.runner_state = dict()

        self.cls_model_manager = ModelManager(configer)
        self.running_score = ClsRunningScore(configer)

        self._init_model()

        if configer.get('data', 'image_tool') == 'pil':
            self.aug_transform = pil_aug_trans.PILAugCompose(configer, split='test')
        elif configer.get('data', 'image_tool') == 'cv2':
            self.aug_transform = cv2_aug_trans.CV2AugCompose(configer, split='test')
        else:
            Log.error('Not support {} image tool.'.format(configer.get('data', 'image_tool')))
            exit(1)

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(**configer.get('data', 'normalize'))])

    def _init_model(self):
        self.cls_net = self.cls_model_manager.get_cls_model()
        self.cls_net = RunnerHelper.load_net(self, self.cls_net)

    def train(self):
        Log.warn('no need')

    def test(self, test_dir, out_dir):
        if test_dir != 'json':
            return self.predict(test_dir, out_dir)

        # don't need 'test_dir' and 'out_dir', only need test.json
        dataset = DefaultDataset(root_dir=self.configer.get('data', 'data_dir'), dataset='test',
                                aug_transform=self.aug_transform, img_transform=self.img_transform,
                                configer=self.configer)

        testloader = data.DataLoader(
            dataset, sampler=None,
            batch_size=self.configer.get('val', 'batch_size'), shuffle=False,
            num_workers=self.configer.get('data', 'workers'), pin_memory=True,
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('test', 'data_transformer')
            ))

        targets_list = []
        predicted_list = []
        path_list = []
        self.cls_net.eval() # keep BN and Dropout
        with torch.no_grad():
            for j, data_dict in enumerate(testloader):
                data_dict = RunnerHelper.to_device(self, data_dict)
                out = self.cls_net(data_dict['img'])
                self.running_score.update({'out': out}, {'out': data_dict['label']})
                targets_list.append(data_dict['label'].cpu())
                predicted_list.append(out.max(1)[1].cpu())
                path_list.extend(data_dict['path'])

                if j == 0:
                    self.images = data_dict['img'].detach()

            top1 = RunnerHelper.dist_avg(self, self.running_score.get_top1_acc())
            top3 = RunnerHelper.dist_avg(self, self.running_score.get_top3_acc())
            top5 = RunnerHelper.dist_avg(self, self.running_score.get_top5_acc())
            if isinstance(top1, dict) and 'out' in top1.keys():
                top1 = top1['out']
                top3 = top3['out']
                top5 = top5['out']
            Log.info('Top1 ACC = {}'.format(top1))
            Log.info('Top3 ACC = {}'.format(top3))
            Log.info('Top5 ACC = {}'.format(top5))

        targets, predicted = torch.cat(targets_list), torch.cat(predicted_list)
        print(confusion_matrix(targets, predicted))
        print(precision_recall_fscore_support(targets, predicted, average='macro'))
        RunnerStat.evaluate(self, targets, predicted, path_list)

    def predict(self, test_dir, out_dir):
        dataset = DirListDataset(root_dir=test_dir, dataset=None,
                                aug_transform=self.aug_transform, img_transform=self.img_transform,
                                configer=self.configer)

        predictloader = data.DataLoader(
            dataset, sampler=None,
            batch_size=self.configer.get('val', 'batch_size'), shuffle=False,
            num_workers=self.configer.get('data', 'workers'), pin_memory=True,
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('test', 'data_transformer')
            ))
            
        predicted_list = []
        path_list = []
        self.cls_net.eval() # keep BN and Dropout
        with torch.no_grad():
            for j, data_dict in enumerate(predictloader):
                out = self.cls_net(data_dict['img'])
                print(F.softmax(out, dim=1))
                predicted_list.append(out.max(1)[1].cpu())
                path_list.extend(data_dict['path'])
