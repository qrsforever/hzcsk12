#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Dataset for Image Classification.

from __future__ import absolute_import, division, print_function

import json
import os

import torch.utils.data as data
import numpy as np
from sklearn.model_selection import train_test_split
from cauchy.extensions.tools.parallel import DataContainer
from cauchy.utils.helpers.image_helper import ImageHelper
from cauchy.utils.tools.logger import Logger as Log


class ClsDataset(data.Dataset):
    def __init__(
        self,
        root_dir=None,
        aug_transform=None,
        img_transform=None,
        configer=None,
        phase=None,
    ):
        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.configer = configer
        self._train_test_split()
        self.root_dir = root_dir
        self.img_list, self.label_list = self.__read_json_file()
        self.aug_transform = aug_transform
        self.img_transform = img_transform

    def _train_test_split(self):
        data_root = self.configer.get("data", "data_dir")
        test_size = self.configer.get("data", "test_size")
        with open(os.path.join(data_root, "train_val.json"), "r") as fin:
            train_val_dict = json.load(fin)
            train_split, test_split = train_test_split(
                train_val_dict, test_size=test_size
            )
            for split, data_split in zip(
                ["train.json", "val.json"], [train_split, test_split]
            ):
                with open(os.path.join(data_root, split), "w") as fout:
                    json.dump(data_split, fout)

    def __getitem__(self, index):
        img = ImageHelper.read_image(
            self.img_list[index],
            tool=self.configer.get("data", "image_tool"),
            mode=self.configer.get("data", "input_mode"),
        )
        label = self.label_list[index]
        img_path = self.img_list[index].replace(self.root_dir, "")

        if self.aug_transform is not None:
            img = self.aug_transform(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(
            img=DataContainer(img, stack=True),
            label=DataContainer(label, stack=True),
            img_path=DataContainer(img_path, stack=True),
        )

    def __len__(self):

        return len(self.img_list)

    def __read_json_file(self):
        img_list = list()
        label_list = list()
        label_file_path = "{0}.json".format(self.phase)

        with open(
            os.path.join(self.root_dir, label_file_path), "r"
        ) as file_stream:
            items = json.load(file_stream)
            for item in items:
                img_list.append(os.path.join(self.root_dir, item["image_path"]))
                label_list.append(item["label"])

        return img_list, label_list
