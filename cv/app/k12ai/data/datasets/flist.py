#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file flist.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-19 17:59

import torch.utils.data as data
from torchvision import transforms

from PIL import Image


class ImageListFileDataset(data.Dataset):
    def __init__(self, flist, labels=None, resize=None, transform=None):
        self.image_list = flist
        self.label_list = labels
        self.resize = resize
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.image_list[index]).convert('RGB')
        if self.resize:
            img = img.resize(self.resize)
        if self.transform is not None:
            img = self.transform(img)
        if self.label_list is not None:
            target = self.label_list[index]
        else:
            target = 0
        return img, target, self.image_list[index]

    def __len__(self):
        return len(self.image_list)
