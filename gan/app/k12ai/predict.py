#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file test.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-01-22 13:23

import time, os
import base64
import tempfile
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
from k12ai.common.log_message import MessageMetric, MessageReport
from k12ai.common.util_misc import print_options


def main(opt):
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.model = 'test'
    if opt.model_suffix == '':
        opt.model_suffix = '_A'
    opt.dataset_mode = 'single'
    print_options(opt)

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    mm = MessageMetric()
    if opt.eval:
        model.eval()
    util.mkdir(opt.results_dir)
    for i, data in enumerate(dataset):
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
            mm.add_image('评估', f'{img_name}', img_path).send()


if __name__ == '__main__':
    opt = TestOptions().parse()
    MessageReport.status(MessageReport.RUNNING)
    try:
        opt.name = opt.dataroot.split('/')[-1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            imgpath = os.path.join(tmp_dir, 'b4img.png')
            with open(imgpath, 'wb') as fw:
                fw.write(base64.b64decode(opt.b64_image))
            opt.dataroot = tmp_dir
            main(opt)
        MessageReport.status(MessageReport.FINISH)
    except Exception:
        MessageReport.status(MessageReport.EXCEPT)
        time.sleep(1)
    finally:
        pass
