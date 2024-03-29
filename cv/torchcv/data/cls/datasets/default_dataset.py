#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Data loader for Image Classification.


import json
import os
import torch.utils.data as data

from lib.parallel.data_container import DataContainer
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.util.logger import Logger as Log


class DefaultDataset(data.Dataset):

    def __init__(self, root_dir=None, dataset=None, aug_transform=None, img_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.img_list, self.label_list = self.read_data(root_dir, dataset)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        label = self.label_list[index]

        if self.aug_transform is not None:
            img = self.aug_transform(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(
            img=DataContainer(img, stack=True),
            label=DataContainer(label, stack=True),
            path=DataContainer(self.img_list[index], stack=True), # QRS: add
        )

    def __len__(self):

        return len(self.img_list)

    def read_data(self, root_dir, dataset):
        img_list = list()
        label_list = list()

        with open(os.path.join(root_dir, '{}.json'.format(dataset)), 'r') as file_stream:
            items = json.load(file_stream)
            for item in items:
                # QRS: fix
                img_path = os.path.join(root_dir, item['image_path'])
                if not os.path.exists(img_path):
                    Log.warn('Image Path: {} not exists.'.format(img_path))
                    continue

                img_list.append(img_path)
                label_list.append(item['label'])

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            with open(os.path.join(root_dir, 'val.json'), 'r') as file_stream:
                items = json.load(file_stream)
                for item in items:
                    # QRS: fix
                    img_path = os.path.join(root_dir, item['image_path'])
                    if not os.path.exists(img_path):
                        Log.warn('Image Path: {} not exists.'.format(img_path))
                        continue

                    img_list.append(img_path)
                    label_list.append(item['label'])
        print(f'Dataset for {dataset} Count: {len(label_list)}')
        return img_list, label_list


class ListDirDataset(DefaultDataset):
    def read_data(self, root_dir, dataset):
        image_list = []
        label_list = []
        image_path = os.path.join(root_dir, dataset)
        if os.path.exists(image_path):
            for item in os.scandir(image_path):
                if item.is_dir():
                    label = int(item.name)
                    for entry in os.scandir(item.path):
                        if entry.is_file():
                            label_list.append(label)
                            image_list.append(entry.path)

        print(f'Dataset for {dataset} Count: {len(label_list)}')
        return image_list, label_list


if __name__ == "__main__":
    # Test data loader.
    pass
