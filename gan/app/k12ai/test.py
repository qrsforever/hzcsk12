#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file test.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-01-21 16:46

import time, os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
from k12ai.common.log_message import MessageMetric, MessageReport


def main(opt):
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    mm = MessageMetric()
    if opt.eval:
        model.eval()
    util.mkdir(opt.results_dir)
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        img_paths = model.get_image_paths()
        visuals = model.get_current_visuals()
        file_name = os.path.splitext(os.path.basename(img_paths[0]))[0]
        path_prefix = os.path.join(opt.results_dir, file_name)
        for img_name, img_data in visuals.items():
            img_path = f'{path_prefix}_{img_name}.png'
            img_nump = util.tensor2im(img_data)
            util.save_image(img_nump, img_path, opt.aspect_ratio)
            if i < 3:
                mm.add_image('评估', f'{path_prefix}_{img_name}', img_path).send()


if __name__ == '__main__':
    opt = TestOptions().parse()
    MessageReport.status(MessageReport.RUNNING)
    try:
        main(opt)
        MessageReport.status(MessageReport.FINISH)
    except Exception:
        MessageReport.status(MessageReport.EXCEPT)
        time.sleep(1)
    finally:
        pass
