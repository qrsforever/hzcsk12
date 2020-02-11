#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file data_loader.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 21:30


class DataLoader:
    def __init__(self, configer):
        self._configer = configer

    def get_dataset(self):
        if 'sklearn' == self._configer.get('data.type'):
            from k12ml.data.sklearn_dataset import sk_get_dataset
            return sk_get_dataset(
                    self._configer.get('data.dataset'),
                    self._configer.get('data.sampling')
            )
        else:
            raise NotImplementedError
