#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file flist.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-19 17:59

import torch.utils.data as data

from PIL import Image


class ImageFilelist(data.Dataset):
    def __init__(self, flist, resize=None, transform=None):
        self.imlist = flist
        self.resize = resize
        self.transform = transform

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = Image.open(impath).convert('RGB')
        if self.resize:
            img.resize(self.resize)
        if self.transform is not None:
                img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imlist)
