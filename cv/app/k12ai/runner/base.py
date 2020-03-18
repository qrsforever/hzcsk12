#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-18 22:28

import torch
import torchvision

from k12ai.common.log_message import MessageMetric

MAX_REPORT_IMGS = 3


class RunnerBase(object):
    def __init__(self):
        self._report_images_num = 0


class ClsRunner(RunnerBase):
    def __init__(self):
        super().__init__()

    def handle_train(self, data_dict):
        mm = MessageMetric()
        iters = self.runner_state['iters']
        if self._report_images_num < MAX_REPORT_IMGS:
            self._report_images_num += 1
            imgs = data_dict['img']
            grid = torchvision.utils.make_grid(imgs.data[:16], nrow=4)
            attr = 'x'.join(map(lambda x: str(x), list(imgs.size())))
            mm.add_image(f'InputImage-{self._report_images_num}', f'{attr}', grid, step=iters)
        mm.send()

    def handle_validation(self):
        pass

    def handle_evaluate(self):
        pass
