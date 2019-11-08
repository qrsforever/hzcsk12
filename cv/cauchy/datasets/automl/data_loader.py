#!/usr/bin/env python
# -*- coding:utf-8 -*-
# dataloader for automl

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Cifar10Loader:
    def __init__(self, configer):
        self.configer = configer
        self.CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        self.CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    def _get_data_transform(self, train=True):
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD),
                ]
            )
            if self.configer.get("data", "cutout"):
                transform.transforms.append(
                    Cutout(self.configer.get("data", "cutout_length"))
                )
            return transform
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD),
                ]
            )
            return transform

    def _get_dataset(self, train=True):
        if train:
            return dset.CIFAR10(
                root=self.configer.get("data", "data_dir"),
                train=True,
                download=True,
                transform=self._get_data_transform(train=True),
            )
        else:
            return dset.CIFAR10(
                root=self.configer.get("data", "data_dir"),
                train=False,
                download=True,
                transform=self._get_data_transform(train=True),
            )

    def get_trainloader(self, search=True):
        if (
            not self.configer.exists("train", "loader")
            or self.configer.get("train", "loader") == "default"
        ):
            dataset = self._get_dataset()
            indices = list(range(50000))
            split = 25000
            if search:
                train_queue = DataLoader(
                    dataset=dataset,
                    batch_size=self.configer.get("train", "batch_size"),
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        indices[:split]
                    ),
                    num_workers=self.configer.get("data", "workers"),
                    pin_memory=True,
                    drop_last=True,
                )
                val_queue = DataLoader(
                    dataset=dataset,
                    batch_size=self.configer.get("train", "batch_size"),
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        indices[split:]
                    ),
                    num_workers=self.configer.get("data", "workers"),
                    pin_memory=True,
                    drop_last=True,
                )
                return train_queue, val_queue
            else:
                return DataLoader(
                    dataset,
                    batch_size=self.configer.get("train", "batch_size"),
                    shuffle=False,
                    drop_last=True,
                )

    def get_valloader(self):
        if (
            not self.configer.exists("val", "loader")
            or self.configer.get("val", "loader") == "default"
        ):
            return DataLoader(
                dataset=self._get_dataset(train=False),
                batch_size=self.configer.get("test", "batch_size"),
                shuffle=True,
                num_workers=self.configer.get("data", "workers"),
                drop_last=True,
            )
