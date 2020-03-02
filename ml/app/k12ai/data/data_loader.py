#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file data_loader.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 21:30

from k12ai.data.datasets import k12ai_get_loader


class DataLoader:
    def __init__(self, configer):
        self._configer = configer

    def get_dataset(self):
        data_dir = self._configer.get('data.data_path')
        if data_dir.startswith('load_'):
            from k12ai.data.sklearn_dataset import sk_get_dataset
            return sk_get_dataset(data_dir, self._configer.get('data.sampling'))

        data_loader = k12ai_get_loader(data_dir.split('/')[-1])
        return data_loader.get_dataset(data_dir, self._configer.get('data.sampling'))
