#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function

import os

from torch.utils import data

import cauchy.datasets.tools.cv2_aug_transforms as cv2_aug_trans
import cauchy.datasets.tools.pil_aug_transforms as pil_aug_trans
import cauchy.datasets.tools.transforms as trans
from cauchy.datasets.tools.collate import collate
from cauchy.utils.tools.logger import Logger as Log

from .dataset import ClsDataset


class DataLoader(object):
    def __init__(self, configer):
        self.configer = configer

        if self.configer.get("data", "image_tool") == "pil":
            self.aug_train_transform = pil_aug_trans.PILAugCompose(
                self.configer, split="train"
            )
        elif self.configer.get("data", "image_tool") == "cv2":
            self.aug_train_transform = cv2_aug_trans.CV2AugCompose(
                self.configer, split="train"
            )
        else:
            Log.error(
                "Not support {} image tool.".format(
                    self.configer.get("data", "image_tool")
                )
            )
            exit(1)

        if self.configer.get("data", "image_tool") == "pil":
            self.aug_val_transform = pil_aug_trans.PILAugCompose(
                self.configer, split="val"
            )
        elif self.configer.get("data", "image_tool") == "cv2":
            self.aug_val_transform = cv2_aug_trans.CV2AugCompose(
                self.configer, split="val"
            )
        else:
            Log.error(
                "Not support {} image tool.".format(
                    self.configer.get("data", "image_tool")
                )
            )
            exit(1)

        self.img_transform = trans.Compose(
            [
                trans.ToTensor(),
                trans.Normalize(**self.configer.get("data", "normalize")),
            ]
        )

    def get_trainloader(self):
        if (
            not self.configer.exists("train", "loader")
            or self.configer.get("train", "loader") == "default"
        ):
            dataset = ClsDataset(
                root_dir=self.configer.get("data", "data_dir"),
                aug_transform=self.aug_train_transform,
                img_transform=self.img_transform,
                configer=self.configer,
                phase="train",
            )
            trainloader = data.DataLoader(
                dataset=dataset,
                batch_size=self.configer.get("train", "batch_size"),
                shuffle=True,
                num_workers=self.configer.get("data", "workers"),
                pin_memory=True,
                drop_last=self.configer.get("data", "drop_last"),
                collate_fn=lambda *args: collate(
                    *args,
                    trans_dict=self.configer.get("train", "data_transformer")
                ),
            )

            return trainloader

        else:
            Log.error(
                "Method: {} loader is invalid.".format(
                    self.configer.get("method")
                )
            )
            return None

    def get_valloader(self):
        if (
            not self.configer.exists("val", "loader")
            or self.configer.get("val", "loader") == "default"
        ):
            # if self.configer.get('method') == 'fc_classifier':
            dataset = ClsDataset(
                root_dir=self.configer.get("data", "data_dir"),
                aug_transform=self.aug_train_transform,
                img_transform=self.img_transform,
                configer=self.configer,
                phase="val",
            )
            valloader = data.DataLoader(
                dataset=dataset,
                batch_size=self.configer.get("val", "batch_size"),
                shuffle=False,
                num_workers=self.configer.get("data", "workers"),
                pin_memory=True,
                drop_last=self.configer.get("data", "drop_last"),
                collate_fn=lambda *args: collate(
                    *args,
                    trans_dict=self.configer.get("val", "data_transformer")
                ),
            )

            return valloader

        else:
            Log.error(
                "Method: {} loader is invalid.".format(
                    self.configer.get("method")
                )
            )
            return None

    def get_testloader(self, test_dir):
        # if self.configer.get('method') == 'fc_classifier':
        if (
            not self.configer.exists("test", "loader")
            or self.configer.get("test", "loader") == "default"
        ):
            dataset = ClsDataset(
                root_dir=test_dir,
                aug_transform=None,
                img_transform=self.img_transform,
                configer=self.configer,
                phase="test",
            )
            testloader = data.DataLoader(
                dataset=dataset,
                batch_size=self.configer.get("test", "batch_size"),
                shuffle=False,
                num_workers=self.configer.get("data", "workers"),
                pin_memory=True,
                drop_last=self.configer.get("data", "drop_last"),
                collate_fn=lambda *args: collate(
                    *args,
                    trans_dict=self.configer.get("val", "data_transformer")
                ),
            )

            return testloader

        else:
            Log.error(
                "Method: {} loader is invalid.".format(
                    self.configer.get("method")
                )
            )
            return None
