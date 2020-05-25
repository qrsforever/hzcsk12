#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-18 22:28

import os
import sys
import math
import torch
import torchvision # noqa
import numpy as np
import datetime

from torchvision import transforms
from PIL import Image
from lib.runner.runner_helper import RunnerHelper # noqa

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from k12ai.data.datasets.flist import ImageListFileDataset
from k12ai.common.log_message import MessageMetric, MessageReport
from k12ai.common.util_misc import transform_denormalize

from k12ai.common.vis_helper import (
    VanillaBackpropagation,
    GuidedBackpropagation,
    Deconvnet, GradCAM,
    FilterFeatureMaps,
    generate_model_graph,
    generate_model_autograd)

MAX_REPORT_IMGS = 2
NUM_MKGRID_IMGS = 16

MAX_CONV2D_HIST = 10


def _load_images(imglist, resize=None, mean=None, std=None):
    raw_images = []
    images = []
    for path in imglist:
        image = Image.open(path).convert('RGB')
        if resize:
            image = image.resize(resize)
        raw_images.append(np.array(image).astype(np.uint8))

        image = transforms.ToTensor()(image)
        if mean and std:
            image = transforms.Normalize(mean=mean, std=std)(image)
        images.append(image)
    return images, raw_images


class RunnerBase(object):
    def __init__(self, data_dir, runner):
        self._mm = MessageMetric()
        self._data_dir = data_dir
        self._report_images_num = 0
        self._cur_epoch = 0
        self._cur_iters = 0
        self._val_batch_time = 0
        self._val_interval = runner.configer.get('solver.test_interval')
        self._max_epoch = runner.configer.get('solver.max_epoch')
        self._max_iters = None
        self._backbone = runner.configer.get('network.backbone')
        self._m_raw_aug = runner.configer.get('metrics.raw_vs_aug', default=False)
        self._m_train_lr = runner.configer.get('metrics.train_lr', default=False)
        self._m_train_remain_time = runner.configer.get('metrics.train_remain_time', default=True)
        self._m_train_loss = runner.configer.get('metrics.train_loss', default=True)
        self._m_train_speed = runner.configer.get('metrics.train_speed', default=False)
        self._m_val_loss = runner.configer.get('metrics.val_loss', default=True)
        self._m_val_speed = runner.configer.get('metrics.val_speed', default=False)

    def send(self):
        self._mm.send()

    def check_loss(self, loss):
        if math.isnan(loss) or loss == float('inf') or loss == float('-inf'):
            MessageReport.status(MessageReport.ERROR, {'err_type': 'LossNanError', 'err_text': 'inf or nan'})
            sys.exit(0)

    def handle_train(self, runner, ddata):
        self._cur_iters = runner.runner_state['iters']
        self._cur_epoch = runner.runner_state['epoch']

        # remain time
        if self._m_train_remain_time:
            batch_time = runner.batch_time.avg
            if self._max_iters is None:
                self._max_iters = self._max_epoch * len(runner.train_loader) # ignore the last epoch
            if self._val_batch_time > 0:
                left_iters = self._max_iters - self._cur_iters
                left_time = batch_time * left_iters + self._val_batch_time * (left_iters // self._val_interval + 1)
                formatted_time = str(datetime.timedelta(seconds=int(left_time)))
                self._mm.add_text('train', 'remain_time', f'{formatted_time}')

        if self._m_raw_aug and self._report_images_num < MAX_REPORT_IMGS:
            self._report_images_num += 1
            imgs, paths = ddata['img'], ddata['path']

            attr = 'x'.join(map(lambda x: str(x), list(imgs.size()[1:])))
            size = (imgs.size(2), imgs.size(3))

            # raw image
            raw_image, _, _ = next(iter(ImageListFileDataset(paths, resize=size)))

            # aug image
            aug_image = transform_denormalize(imgs.data[0].cpu(), **runner.configer.get('data', 'normalize'))

            grid_image = torchvision.utils.make_grid([raw_image, aug_image])
            self._mm.add_image('image', f'{self._report_images_num}-{attr}', grid_image)

        # learning rate
        if self._m_train_lr:
            self._mm.add_scalar('train', 'learning_rate', x=self._cur_iters, y=runner.optimizer.param_groups[0]['lr'])

        # batch time: speed
        if self._m_train_speed:
            self._mm.add_scalar('train', 'speed', x=self._cur_iters, y=1.0 / batch_time)

    def handle_validation(self, runner):
        self._cur_iters = runner.runner_state['iters']
        self._cur_epoch = runner.runner_state['epoch']

        batch_time = runner.batch_time.avg
        self._val_batch_time = runner.batch_time.sum

        # batch time: speed
        if self._m_val_speed:
            self._mm.add_scalar('val', 'speed', x=self._cur_iters, y=1.0 / batch_time)

        # progress
        self._mm.add_scalar('train_val', 'progress', x=self._cur_iters, y=round(100 * self._cur_epoch / self._max_epoch, 2))

    def handle_evaluate(self, runner):
        pass


class ClsRunner(RunnerBase):
    def __init__(self, data_dir, runner):
        super().__init__(data_dir, runner)
        self._model = runner.cls_net

    def handle_train(self, runner, ddata):
        super().handle_train(runner, ddata)

        # loss
        if self._m_train_loss:
            loss = list(runner.train_losses.avg.values())[0]
            self.check_loss(loss)
            self._mm.add_scalar('train', 'loss', x=self._cur_iters, y=loss)

        return self

    def handle_validation(self, runner):
        super().handle_validation(runner)
        # loss: train and val
        if self._m_val_loss:
            y = {
                'train': list(runner.train_losses.avg.values())[0],
                'val': list(runner.val_losses.avg.values())[0]
            }
            self._mm.add_scalar('train_val', 'loss', x=self._cur_iters, y=y)

        # acc
        y = {
            'top1': runner.running_score.get_top1_acc()['out'],
            'top3': runner.running_score.get_top3_acc()['out'],
            'top5': runner.running_score.get_top5_acc()['out']
        }
        self._mm.add_scalar('val', 'acc', x=self._cur_iters, y=y)

        return self

    def handle_evaluate(self, runner, y_true, y_pred, files):
        super().handle_evaluate(runner)

        # acc
        top1 = runner.running_score.get_top1_acc()['out']
        top3 = runner.running_score.get_top3_acc()['out']
        top5 = runner.running_score.get_top5_acc()['out']
        self._mm.add_text('result', 'acc', f'{top1, top3, top5}')

        # confusion matrix
        if runner.configer.get('metrics.confusion_matrix', default=False):
            if runner.configer.get('data.num_classes') < 100:
                cm = confusion_matrix(y_true, y_pred)
                self._mm.add_matrix('measuring', 'confusion_matrix', cm).send()

        # 10 images
        if runner.configer.get('metrics.top10_images', default=False):
            for i, (true, pred, path) in enumerate(zip(y_true, y_pred, files)):
                if i >= 10:
                    break
                self._mm.add_image('top10_images', f'IMG-{i+1}_{true}_vs_{pred}', path).send()

        # 10 error images
        if runner.configer.get('metrics.top10_errors', default=False):
            i = 0
            for true, pred, path in zip(y_true, y_pred, files):
                if i >= 10:
                    break
                if true != pred:
                    self._mm.add_image('top10_errors', f'{os.path.basename(path)}_{true}_vs_{pred}', path).send()
                    i += 1

        # precision, recall, fscore
        P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        if runner.configer.get('metrics.precision', default=False):
            self._mm.add_text('measuring', 'precision', P)
        if runner.configer.get('metrics.recall', default=False):
            self._mm.add_text('measuring', 'recall', R)
        if runner.configer.get('metrics.fscore', default=False):
            self._mm.add_text('measuring', 'fscore', F)
        self._mm.send()

        # model graph
        if runner.configer.get('metrics.model_autograd', default=False):
            value = generate_model_autograd(self._model.module, runner.images[0:1], fmt='svg')
            self._mm.add_image('model', 'autograd', value, fmt='svg')
            self._mm.send()

        if runner.configer.get('metrics.model_graph', default=False):
            value = generate_model_graph(self._model.module, runner.images[0:1], fmt='svg')
            self._mm.add_image('model', 'forword', value, fmt='svg')
            self._mm.send()

        images, raw_images = _load_images(files[0:1], runner.images.shape[2:])

        # Vanilla Backpropagation Saliency Map
        if runner.configer.get('metrics.vbp', default=False):
            vbp = VanillaBackpropagation(self._model.module)
            probs, ids_sorted = vbp.forward(images)
            vbp.backward(ids=ids_sorted[:, [0]])
            gradients = vbp.generate()
            image_numpy = vbp.mkimage(gradients[0])
            self._mm.add_image('cnn_heat_maps', 'vanillabackpropagation', image_numpy)
            self._mm.send()
            vbp.remove_hook()

        # Deconvnet
        if runner.configer.get('metrics.deconv', default=False):
            deconv = Deconvnet(self._model.module)
            probs, ids_sorted = deconv.forward(images)
            deconv.backward(ids=ids_sorted[:, [0]])
            gradients = deconv.generate()
            image_numpy = deconv.mkimage(gradients[0])
            self._mm.add_image('cnn_heat_maps', 'deconvnet', image_numpy)
            self._mm.send()
            deconv.remove_hook()

        # Guided Backpropagation
        if runner.configer.get('metrics.gbp', default=False):
            gbp = GuidedBackpropagation(self._model.module)
            probs, ids_sorted = gbp.forward(images)
            gbp.backward(ids=ids_sorted[:, [0]])
            gradients = gbp.generate()
            image_numpy = gbp.mkimage(gradients[0])
            self._mm.add_image('cnn_heat_maps', 'guidedbackpropagation', image_numpy)
            self._mm.send()
            gbp.remove_hook()

        # G-CAM
        if runner.configer.get('metrics.gcam', default=False):
            if self._backbone.startswith('resnet'):
                target_layer = 'net.layer4'
            elif self._backbone.startswith('vgg'):
                target_layer = 'net.features'
            elif self._backbone.startswith('alexnet'):
                target_layer = 'net.features'
            else:
                target_layer = None
            if target_layer:
                gcam = GradCAM(self._model.module, [target_layer])
                probs, ids_sorted = gcam.forward(images)
                gcam.backward(ids=ids_sorted[:, [0]])
                regions = gcam.generate(target_layer)
                image_numpy = gcam.mkimage(regions[0, 0], raw_images[0])
                self._mm.add_image('cnn_heat_maps', 'g-cam', image_numpy)
                self._mm.send()
                gcam.remove_hook()

        # Guilded G-CAM
        if runner.configer.get('metrics.ggcam', default=False):
            if self._backbone.startswith('resnet'):
                target_layer = 'net.layer4'
            elif self._backbone.startswith('vgg'):
                target_layer = 'net.features'
            elif self._backbone.startswith('alexnet'):
                target_layer = 'net.features'
            else:
                target_layer = None
            if target_layer:
                # Guilded Backprogation
                gbp = GuidedBackpropagation(self._model.module)
                probs, ids_sorted = gbp.forward(images)
                gbp.backward(ids=ids_sorted[:, [0]])
                gradients = gbp.generate()

                gcam = GradCAM(self._model.module, [target_layer])
                probs, ids_sorted = gcam.forward(images)
                gcam.backward(ids=ids_sorted[:, [0]])
                regions = gcam.generate(target_layer)

                image_numpy = gbp.mkimage(torch.mul(regions, gradients)[0])
                self._mm.add_image('cnn_heat_maps', 'guided_g-cam', image_numpy)
                self._mm.send()
                gcam.remove_hook()

        # Feature Maps, Filters
        feature_maps = runner.configer.get('metrics.feature_maps', default=False)
        filters_maps = runner.configer.get('metrics.filters_maps', default=False)
        if feature_maps or filters_maps:
            target_layers = []
            for name, module in self._model.module.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layers.append(name)
            fm = FilterFeatureMaps(self._model.module, target_layers[:5], with_filter=filters_maps)
            fm.forward(images)
            image_grid, filter_grids = fm.generate(rank='TB')
            image_bytes = fm.mkimage(image_grid)
            self._mm.add_image('cnn_heat_maps', 'feature_maps', image_bytes)
            self._mm.send()
            for i, (image_grid, layer_name, shape) in enumerate(filter_grids):
                attr = 'x'.join(map(lambda x: str(x), list(shape)))
                self._mm.add_image('cnn_heat_maps', f'filter_{layer_name}_{attr}', fm.mkimage(image_grid))
                self._mm.send()
            fm.remove_hook()

        return self


class DetRunner(RunnerBase):
    def __init__(self, data_dir, runner):
        super().__init__(data_dir, runner)
        self._model = runner.det_net

    def handle_train(self, runner, ddata):
        super().handle_train(runner, ddata)

        # loss
        loss = runner.train_losses.val
        self.check_loss(loss)
        self._mm.add_scalar('train', 'loss', x=self._cur_iters, y=loss)

        return self

    def handle_validation(self, runner):
        super().handle_validation(runner)
        # loss.avg: train and val
        y = {
            'train': runner.train_losses.avg,
            'val': runner.val_losses.avg
        }
        self._mm.add_scalar('train_val', 'loss', x=self._cur_iters, y=y)

        # mAP
        self._mm.add_scalar('val', 'mAP', x=self._cur_iters, y=runner.det_running_score.get_mAP())

        return self

    def handle_evaluate(self, runner):
        super().handle_evaluate(runner)
        # mAP
        self._mm.add_text('evaluate', 'mAP', runner.det_running_score.get_mAP())

        return self


class RunnerStat(object):
    H = None

    @staticmethod
    def check(runner):
        if RunnerStat.H is None:
            data_dir = runner.configer.get('data', 'data_dir')
            task = runner.configer.get('task')
            if task == 'cls':
                RunnerStat.H = ClsRunner(data_dir, runner)
            elif task == 'det':
                RunnerStat.H = DetRunner(data_dir, runner)
            else:
                raise NotImplementedError

    @staticmethod
    def train(runner, ddata):
        RunnerStat.check(runner)
        RunnerStat.H.handle_train(runner, ddata).send()

    @staticmethod
    def validation(runner):
        RunnerStat.check(runner)
        RunnerStat.H.handle_validation(runner).send()

    @staticmethod
    def evaluate(runner, *args, **kwargs):
        RunnerStat.check(runner)
        RunnerStat.H.handle_evaluate(runner, *args, **kwargs).send()
