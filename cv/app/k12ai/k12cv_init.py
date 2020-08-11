#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_init.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-02 19:38:04

import os

from lib.tools.util.logger import Logger as Log

from k12ai.runner.cls.image_classifier_test import ImageClassifierTest
from k12ai.runner.det.single_shot_detector_test import SingleShotDetectorTest
from k12ai.model.det.nets.custom_ssd300 import CustomSSD300
from k12ai.model.det.nets.custom_ssd512 import CustomSSD512
from k12ai.model.cls.nets.custom_base import CustomBaseModel
from k12ai.tools.util.net_def import build_custom_model

from runner.runner_selector import CLS_TEST_DICT, DET_TEST_DICT
from model.det.model_manager import DET_MODEL_DICT
from model.cls.model_manager import CLS_MODEL_DICT

PRETRAINED_MODELS = {
    'vgg11': 'vgg11-bbd30ac9.pth',
    'vgg13': 'vgg13-c768596a.pth',
    'vgg16': 'vgg16-397923af.pth',
    'vgg19': 'vgg19-dcbb9e9d.pth',
    'alexnet': 'alexnet-owt-4df8aa71.pth',
    'vgg11_bn': 'vgg11_bn-6002323d.pth',
    'vgg13_bn': 'vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'vgg19_bn-c79401a0.pth',
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
    'densenet121': 'densenet121-a639ec97.pth',
    'densenet169': 'densenet169-b2777c0a.pth',
    'densenet201': 'densenet201-c1103571.pth',
    'densenet161': 'densenet161-8d451a50.pth',
    'mobilenet_v2': 'mobilenet_v2-b0353104.pth',
    'squeezenet1_0': 'squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'squeezenet1_1-f364aa15.pth',
    'shufflenetv2_x0.5': 'shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'shufflenetv2_x1-5666bf0f80.pth',
}


# change the original method to hzcsk12 hook
def _hook_runner_selector(configer, custom):
    task = configer.get('task')
    Log.info("_hook_runner_selector(%s)" % task)
    if task == 'cls':
        if custom:
            CLS_MODEL_DICT['custom_base'] = CustomBaseModel
        CLS_TEST_DICT['image_classifier'] = ImageClassifierTest
    elif task == 'det':
        if custom:
            DET_MODEL_DICT['custom_ssd300'] = CustomSSD300
            DET_MODEL_DICT['custom_ssd512'] = CustomSSD512
        DET_TEST_DICT['single_shot_detector'] = SingleShotDetectorTest
    else:
        Log.info("not impl")


# build custom model if model name begin with the prefix 'custom_'
def _check_custom_model(configer):
    model_name = configer.get('network.model_name')
    Log.info('model name[%s]' % model_name)
    if model_name.split("_")[0] == "custom":
        if model_name not in (
                'custom_base',
                'custom_ssd300',
                'custom_ssd512'):
            raise RuntimeError('Model: {} not valid!'.format(model_name))
        try:
            cache_dir = configer.get('network.checkpoints_root')
            net_def_str = configer.get('network.net_def')
            net_def_dir = build_custom_model(cache_dir, net_def_str, model_name)
            net_def_fil = os.path.join(net_def_dir, '%s.txt' % model_name)
            with open(net_def_fil, 'w') as fout:
                fout.write(net_def_str)
        except Exception as err:
            print('err: {}'.format(err))
            raise RuntimeError('{}'.format(err))
        return True
    return False


def _verify_config(backbone, configer):
    if 'alexnet' == backbone:
        for phase in ('train', 'val', 'test'):
            inputsize = configer.get(f'{phase}.data_transformer.input_size')
            if inputsize[0] < 64 or inputsize[1] < 64:
                configer.update(f'{phase}.data_transformer.input_size', [64, 64])


def k12ai_cv_init(configer):
    Log.debug('k12ai_cv_init')

    # Check
    # if configer.get('phase') == 'test':
    #     configer.update('network.resume_continue', True)
    # if configer.get('network.resume_continue'):
    #     ck_root = configer.get('network.checkpoints_root')
    #     ck_dir = configer.get('network.checkpoints_dir')
    #     ck_name = configer.get('network.checkpoints_name')
    #     configer.update('network.resume', f'{ck_root}/{ck_dir}/{ck_name}_latest.pth')

    # metric = configer.get('solver.lr.metric')
    # if metric == 'epoch':
    #     max_epoch = configer.get('solver.max_epoch')
    #     Log.info('_k12ai.solver.lr.metric: epoch, max: %d' % max_epoch)
    # else:
    #     max_iters = configer.get('solver.max_iters')
    #     Log.info('_k12ai.solver.lr.metric: iters, max: %d' % max_iters)

    backbone = configer.get('network.backbone', default='unknow')

    # Verify Backbone
    _verify_config(backbone, configer)

    # Pretrained
    resume_continue = configer.get('network.resume_continue')
    if resume_continue:
        configer.update('network.pretrained', None)
    pretrained = configer.get('network.pretrained')
    configer.update('network.pretrained', None)
    if pretrained:
        pretrained_file = PRETRAINED_MODELS.get(backbone, 'nofile')
        if os.path.exists(f'/pretrained/{pretrained_file}'):
            configer.update('network.pretrained', f'/pretrained/{pretrained_file}')

    # Custom
    custom = _check_custom_model(configer)

    _hook_runner_selector(configer, custom)
