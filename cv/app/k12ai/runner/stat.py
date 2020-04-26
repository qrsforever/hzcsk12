#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-18 22:28

import sys
import math
import torch
import torchvision # noqa
import numpy as np

from torchvision import transforms
from PIL import Image
from lib.runner.runner_helper import RunnerHelper # noqa

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from k12ai.data.datasets.flist import ImageListFileDataset
from k12ai.common.log_message import MessageMetric, MessageReport
from k12ai.common.util_misc import (
    transform_denormalize,
    generate_model_graph,
    generate_model_autograd)

from k12ai.common.vis_helper import GradCAM

MAX_REPORT_IMGS = 2
NUM_MKGRID_IMGS = 16

MAX_CONV2D_HIST = 10


def _load_images(imglist, resize, mean, std):
    raw_images = []
    images = []
    for path in imglist:
        image = Image.open(path).convert('RGB')
        image = np.array(image.resize(resize)).astype(np.uint8)
        raw_images.append(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=mean, std=std)(image)
        images.append(image)
    return images, raw_images


class RunnerBase(object):
    def __init__(self, data_dir, runner):
        self._mm = MessageMetric()
        self._data_dir = data_dir
        self._report_images_num = 0
        self._epoch = 0
        self._iters = 0
        self._max_epoch = runner.configer.get('solver.max_epoch')
        self._backbone = runner.configer.get('network.backbone')

    def send(self):
        self._mm.send()

    def check_loss(self, loss):
        if math.isnan(loss) or loss == float('inf') or loss == float('-inf'):
            MessageReport.status(MessageReport.ERROR, {'err_type': 'LossNanError', 'err_text': 'inf or nan'})
            sys.exit(0)

    def handle_train(self, runner, ddata):
        self._iters = runner.runner_state['iters']
        self._epoch = runner.runner_state['epoch']

        if self._report_images_num < MAX_REPORT_IMGS:
            self._report_images_num += 1
            imgs, paths = ddata['img'], ddata['path']

            attr = 'x'.join(map(lambda x: str(x), list(imgs.size()[1:])))
            size = (imgs.size(2), imgs.size(3))

            # raw image
            raw_image, _, _ = next(iter(ImageListFileDataset(paths, resize=size)))
            self._mm.add_image('image_transform', f'RAW-{self._report_images_num}-{attr}', raw_image)

            # aug image
            aug_image = transform_denormalize(imgs.data[0].cpu(), **runner.configer.get('data', 'normalize'))
            self._mm.add_image('image_transform', f'AUG-{self._report_images_num}-{attr}', aug_image)

        # learning rate
        self._mm.add_scalar('train', 'learning_rate', x=self._iters, y=runner.optimizer.param_groups[0]['lr'])

        # batch time: speed
        self._mm.add_scalar('train', 'speed', x=self._iters, y=1.0 / runner.batch_time.avg)

    def handle_validation(self, runner):
        self._iters = runner.runner_state['iters']
        self._epoch = runner.runner_state['epoch']

        # batch time: speed
        self._mm.add_scalar('val', 'speed', x=self._iters, y=1.0 / runner.batch_time.avg)

        # progress
        self._mm.add_scalar('train_val', 'progress', x=self._iters, y=round(100 * self._epoch / self._max_epoch, 2))

    def handle_evaluate(self, runner):
        pass


class ClsRunner(RunnerBase):
    def __init__(self, data_dir, runner):
        super().__init__(data_dir, runner)
        self._model = runner.cls_net

    def handle_train(self, runner, ddata):
        super().handle_train(runner, ddata)

        # loss
        loss = list(runner.train_losses.avg.values())[0]
        self.check_loss(loss)
        self._mm.add_scalar('train', 'loss', x=self._iters, y=loss)

        return self

    def handle_validation(self, runner):
        super().handle_validation(runner)
        # loss: train and val
        y = {
            'train': list(runner.train_losses.avg.values())[0],
            'val': list(runner.val_losses.avg.values())[0]
        }
        self._mm.add_scalar('train_val', 'loss', x=self._iters, y=y)

        # acc
        y = {
            'top1': runner.running_score.get_top1_acc()['out'],
            'top3': runner.running_score.get_top3_acc()['out'],
            'top5': runner.running_score.get_top5_acc()['out']
        }
        self._mm.add_scalar('val', 'acc', x=self._iters, y=y)

        # weight, bias, grad
        if self._epoch != self._epoch and self._epoch < MAX_CONV2D_HIST:
            for key, module in self._model.named_modules():
                if not isinstance(module, torch.nn.Conv2d):
                    continue
                if module.weight is not None and module.weight.grad.data is not None:
                    self._mm.add_histogram('train', 'conv2d_1_weight', module.weight.data, self._epoch)
                    self._mm.add_histogram('train', 'conv2d_1_weight.grad', module.weight.grad.data, self._epoch)
                if module.bias is not None and module.bias.grad.data is not None:
                    self._mm.add_histogram('train', 'conv2d_1_bias', module.bias.data)
                    self._mm.add_histogram('train', 'conv2d_1_bias.grad', module.bias.grad.data, self._epoch)
                break

        return self

    def handle_evaluate(self, runner, y_true, y_pred, files):
        super().handle_evaluate(runner)

        # confusion matrix
        if runner.configer.get('metrics.confusion_matrix', default=False):
            if runner.configer.get('data.num_classes') < 100:
                cm = confusion_matrix(y_true, y_pred)
                self._mm.add_matrix('confusion_matrix', 'matrix', cm).send()

        # images top10
        if runner.configer.get('metrics.true_pred_images', default=False):
            for i, (true, pred, path) in enumerate(zip(y_true, y_pred, files)):
                if i >= 10:
                    break
                self._mm.add_image('predict_top10', f'IMG-{i+1}_{true}_vs_{pred}', path).send()

        # precision, recall, fscore
        P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        self._mm.add_text('evaluate', 'precision', P)
        self._mm.add_text('evaluate', 'recall', R)
        self._mm.add_text('evaluate', 'fscore', F)
        self._mm.send()

        # model graph
        if runner.configer.get('metrics.model_autograd', default=False):
            value = generate_model_autograd(self._model.module, runner.images[0:1], fmt='svg')
            self._mm.add_image('model_graph', 'autograd', value, fmt='svg')
            self._mm.send()

        if runner.configer.get('metrics.model_graph', default=False):
            value = generate_model_graph(self._model.module, runner.images[0:1], fmt='svg')
            self._mm.add_image('model_graph', 'forword', value, fmt='svg')
            self._mm.send()

        # G-CAM
        if runner.configer.get('metrics.gcam', default=False):
            if self._backbone.startswith('resnet'):
                target_layer = 'net.layer4'
            elif self._backbone.startswith('vgg'):
                target_layer = 'net.features'
            else:
                target_layer = None
            if target_layer:
                images, raw_images = _load_images(files[0:1], runner.images.shape[2:], **runner.configer.get('data', 'normalize'))
                gcam = GradCAM(self._model.module, [target_layer])
                probs, ids_sorted = gcam.forward(images)
                gcam.backward(ids=ids_sorted[:, [0]])
                regions = gcam.generate(target_layer)
                image_numpy = gcam.mkimage(regions[0, 0], raw_images[0])
                self._mm.add_image('gcam', 'gcam0', image_numpy, fmt='svg')
                self._mm.send()

        # Gradient
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

        # image = Image.open(files[0]).convert('RGB')
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean, std),
        # ])
        # image_input = transform(image).unsqueeze(0)

        # def input_gradient_hook(grad):
        #     self.gradient = grad

        # image_input.requires_grad = True
        # image_input.register_hook(input_gradient_hook)

        # data_dict = RunnerHelper.to_device(runner, {'img': image_input, 'label': torch.Tensor([y_true[0]])})
        # output = runner.cls_net(data_dict['img'])
        # runner.cls_net.zero_grad()
        # onehot = torch.FloatTensor(1, output.size()[-1]).zero_()
        # onehot[0][y_true[0]] = 1
        # onehot = RunnerHelper.to_device(runner, onehot)
        # output.backward(gradient=onehot)

        # gradient = self.gradient
        # gradient -= gradient.min()
        # gradient /= gradient.max()

        # grid = torchvision.utils.make_grid(torch.cat((image_input, gradient)), padding=0)

        # self._mm.add_image('gradient', f'input_layer', grid)
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
        self._mm.add_scalar('train', 'loss', x=self._iters, y=loss)

        return self

    def handle_validation(self, runner):
        super().handle_validation(runner)
        # loss.avg: train and val
        y = {
            'train': runner.train_losses.avg,
            'val': runner.val_losses.avg
        }
        self._mm.add_scalar('train_val', 'loss', x=self._iters, y=y)

        # mAP
        self._mm.add_scalar('val', 'mAP', x=self._iters, y=runner.det_running_score.get_mAP())

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
