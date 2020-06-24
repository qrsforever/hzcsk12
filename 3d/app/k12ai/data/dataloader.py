#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file dataloader.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-24 19:55

import os
import torch
import torchvision


AUGMENTATIONS_DICT = {
    "center_crop": torchvision.transforms.CenterCrop,
    "color_jitter": torchvision.transforms.ColorJitter,
    "random_horizontal_flip": torchvision.transforms.RandomHorizontalFlip,
    "resize": torchvision.transforms.Resize
}


class Dataloader(object):
    def __init__(self, configer):
        self.configer = configer
        self.train_trans = None
        self.valid_trans = None
        self.test_trans  = None # noqa
        compose = self.configer.get('transforms.compose', default=None)
        if compose:
            ttrans, vtrans, etrans = [], [], []
            for key, value in compose.items():
                func = AUGMENTATIONS_DICT[key]
                if compose.get(f'{key}.phase.train', default=False):
                    ttrans.append(func(**compose.get(f'{key}.args', default={})))
                if compose.get(f'{key}.phase.valid', default=False):
                    vtrans.append(func(**compose.get(f'{key}.args', default={})))
                if compose.get(f'{key}.phase.evaluate', default=False):
                    etrans.append(func(**compose.get(f'{key}.args', default={})))
            if len(ttrans) > 0:
                self.train_trans = torchvision.transforms.Compose(ttrans)
            if len(vtrans) > 0:
                self.valid_trans = torchvision.transforms.Compose(vtrans)
            if len(etrans) > 0:
                self.test_trans  = torchvision.transforms.Compose(etrans) # noqa

    def get_trainloader(self):
        if 'nyu' == self.configer.get('dataset_name'):
            from .nyu.dataset_reader import NYUDatasetReader
            dataset = NYUDatasetReader(
                    root=os.path.join(self.configer.get('dataset_root'), 'train'),
                    output_size=self.configer.get('transforms.output_size', default=None),
                    normalize=self.configer.get('transforms.normalize', default=None),
                    transforms=self.train_trans)
        else:
            raise NotImplementedError

        return torch.utils.data.DataLoader(dataset, **self.configer.get('dataset_loader.args'))

    def get_validloader(self):
        if 'nyu' == self.configer.get('dataset_name'):
            from .nyu.dataset_reader import NYUDatasetReader
            dataset = NYUDatasetReader(
                    root=os.path.join(self.configer.get('dataset_root'), 'val'),
                    output_size=self.configer.get('transforms.output_size', default=None),
                    normalize=self.configer.get('transforms.normalize', default=None),
                    transforms=self.valid_trans)
        else:
            raise NotImplementedError

        return torch.utils.data.DataLoader(dataset, **self.configer.get('dataset_loader.args'))

    def get_testloader(self):
        if 'nyu' == self.configer.get('dataset_name'):
            from .nyu.dataset_reader import NYUDatasetReader
            dataset = NYUDatasetReader(
                    root=os.path.join(self.configer.get('dataset_root'), 'test'),
                    output_size=self.configer.get('transforms.output_size', default=None),
                    normalize=self.configer.get('transforms.normalize', default=None),
                    transforms=self.test_trans)
        else:
            raise NotImplementedError

        return torch.utils.data.DataLoader(dataset, **self.configer.get('dataset_loader.args'))
