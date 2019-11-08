#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Class for the Lane Pose Loader.

from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import torch
import torch.utils.data as data

from cauchy.extensions.tools.parallel import DataContainer
from cauchy.datasets.pose.utils.heatmap_generator import HeatmapGenerator
from cauchy.utils.helpers.json_helper import JsonHelper
from cauchy.utils.helpers.image_helper import ImageHelper
from cauchy.utils.tools.logger import Logger as Log


class DefaultLoader(data.Dataset):
    def __init__(
        self,
        root_dir,
        dataset=None,
        aug_transform=None,
        img_transform=None,
        configer=None,
    ):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.heatmap_generator = HeatmapGenerator(self.configer)
        (self.img_list, self.json_list) = self.__list_dirs(root_dir, dataset)

    def __getitem__(self, index):
        img = ImageHelper.read_image(
            self.img_list[index],
            tool=self.configer.get("data", "image_tool"),
            mode=self.configer.get("data", "input_mode"),
        )

        kpts, bboxes = self.__read_json_file(self.json_list[index])

        if self.aug_transform is not None:
            img, kpts, bboxes = self.aug_transform(
                img, kpts=kpts, bboxes=bboxes
            )

        kpts = torch.from_numpy(kpts).float()
        heatmap = self.heatmap_generator(kpts, ImageHelper.get_size(img))
        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(
            img=DataContainer(img, stack=True),
            heatmap=DataContainer(heatmap, stack=True),
        )

    def __len__(self):

        return len(self.img_list)

    def __read_json_file(self, json_file):
        """
            filename: JSON file

            return: three list: key_points list, centers list and scales list.
        """
        json_dict = JsonHelper.load_file(json_file)
        kpts = list()
        bboxes = list()

        for object in json_dict["objects"]:
            kpts.append(object["kpts"])
            if "bbox" in object:
                bboxes.append(object["bbox"])

        return (
            np.array(kpts).astype(np.float32),
            np.array(bboxes).astype(np.float32),
        )

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        json_list = list()
        image_dir = os.path.join(root_dir, dataset, "image")
        json_dir = os.path.join(root_dir, dataset, "json")

        img_extension = os.listdir(image_dir)[0].split(".")[-1]

        for file_name in os.listdir(json_dir):
            image_name = ".".join(file_name.split(".")[:-1])
            img_path = os.path.join(
                image_dir, "{}.{}".format(image_name, img_extension)
            )
            json_path = os.path.join(json_dir, file_name)
            if not os.path.exists(json_path) or not os.path.exists(img_path):
                Log.warn("Json Path: {} not exists.".format(json_path))
                continue

            json_list.append(json_path)
            img_list.append(img_path)

        if dataset == "train" and self.configer.get("data", "include_val"):
            image_dir = os.path.join(root_dir, "val/image")
            json_dir = os.path.join(root_dir, "val/json")
            for file_name in os.listdir(json_dir):
                image_name = ".".join(file_name.split(".")[:-1])
                img_path = os.path.join(
                    image_dir, "{}.{}".format(image_name, img_extension)
                )
                json_path = os.path.join(json_dir, file_name)
                if not os.path.exists(json_path) or not os.path.exists(
                    img_path
                ):
                    Log.warn("Json Path: {} not exists.".format(json_path))
                    continue

                json_list.append(json_path)
                img_list.append(img_path)

        return img_list, json_list


if __name__ == "__main__":
    # Test Lane Loader.
    pass
