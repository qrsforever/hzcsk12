#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file dataset_reader.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-21 12:50

import os
import h5py
import numpy as np
import torchvision

from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.h5', ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def listdir_dataset(root):
    images = []
    for target in os.listdir(root):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            if is_image_file(d):
                images.append(d)
    return images


class NYUDatasetReader(Dataset):
    def __init__(self, root, output_size=None, normalize=None, transforms=None):
        self.output_size = output_size
        self.normalize = normalize
        self.transforms = transforms
        self.imgs = listdir_dataset(root)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        with h5py.File(self.imgs[index], 'r') as h5f:
            input_h5 = np.array(h5f['rgb'])
            depth_h5 = np.array(h5f['depth'])
        input_pil = torchvision.transforms.ToPILImage()(input_h5)
        depth_pil = torchvision.transforms.ToPILImage()(depth_h5)

        if self.transforms:
            input_pil = self.transforms(input_pil)
            depth_pil = self.transforms(depth_pil)

        if self.output_size:
            input_pil = torchvision.transforms.Resize(self.output_size)(input_pil)
            depth_pil = torchvision.transforms.Resize(self.output_size)(depth_pil)

        input_tensor = torchvision.transforms.ToTensor()(np.array(input_pil))
        depth_tensor = torchvision.transforms.ToTensor()(np.array(depth_pil))
        if self.normalize:
            mean = self.normalize.get('mean')
            std  = self.normalize.get('std') # noqa
            input_tensor = torchvision.transforms.Normalize(mean, std)(input_tensor)

        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        return input_tensor, depth_tensor
