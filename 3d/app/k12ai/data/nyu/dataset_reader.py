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
    for target in sorted(os.listdir(root)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            if is_image_file(d):
                images.append(d)
    return images


class NYUDatasetReader(Dataset):
    def __init__(self, root, normalize=None, transforms=None):
        self.output_size = (228, 304)
        self.imgs = listdir_dataset(root)
        self.transforms = transforms
        self.normalize = normalize

    def __getraw__(self, index):
        with h5py.File(self.imgs[index], 'r') as h5f:
            rgb = np.array(h5f['rgb'])
            depth = np.array(h5f['depth'])
            return rgb, depth
        return None, None

    def __getitem__(self, index):
        with h5py.File(self.imgs[index], 'r') as h5f:
            input = np.array(h5f['rgb'])
            depth = np.array(h5f['depth'])

        input_pil = torchvision.transforms.ToPILImage()(input)
        depth_pil = torchvision.transforms.ToPILImage()(depth)

        if self.transforms:
            input_pil = self.transforms(input_pil)
            depth_pil = self.transforms(depth_pil)
        input_tensor = torchvision.transforms.ToTensor()(input_pil)
        depth_tensor = torchvision.transforms.ToTensor()(depth_pil)

        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        return input_tensor, depth_tensor

    # def train_transform(self, rgb, depth):
    #     transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize(250),  # this is for computational efficiency, since rotation can be slow
    #         transforms.CenterCrop(self.output_size),
    #         transforms.RandomHorizontalFlip(0.5),
    #     ])
    #     rgb_np = transform(rgb)
    #     rgb_np = transforms.ColorJitter(0.4, 0.4, 0.4)(rgb_np)
    #     rgb_tensor = transforms.ToTensor()(np.array(rgb_np))
    #     depth_np = transform(depth)
    #     depth_tensor = transforms.ToTensor()(np.array(depth_np))

    #     return rgb_tensor, depth_tensor

    # def val_transform(self, rgb, depth):
    #     depth_np = depth
    #     transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize(250),
    #         transforms.CenterCrop(self.output_size),
    #     ])
    #     rgb_np = transform(rgb)
    #     rgb_tensor = transforms.ToTensor()(np.array(rgb_np))
    #     depth_np = transform(depth_np)
    #     depth_tensor = transforms.ToTensor()(np.array(depth_np))

    #     return rgb_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)
