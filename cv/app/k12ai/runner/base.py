#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-18 22:28

import torch # noqa
import torchvision

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from k12ai.data.datasets.flist import ImageFilelist
from k12ai.common.log_message import MessageMetric

MAX_REPORT_IMGS = 1
NUM_MKGRID_IMGS = 4

MAX_CONV2D_HIST = 10


class RunnerBase(object):
    def __init__(self, data_dir, model):
        self._data_dir = data_dir
        self._model = model
        self._report_images_num = 0
        self._epoch = 0

    def handle_train(self, runner, ddata):
        iters = runner.runner_state['iters']

        mm = MessageMetric()
        if self._report_images_num < MAX_REPORT_IMGS:
            self._report_images_num += 1
            imgs, labels, paths = ddata['img'], ddata['label'], ddata['path']

            # aug image
            grid = torchvision.utils.make_grid(imgs.data[:NUM_MKGRID_IMGS], nrow=8, padding=0)
            attr = 'x'.join(map(lambda x: str(x), list(imgs.size())))
            mm.add_image('train', f'Aug-{attr}-{self._report_images_num}', grid)

            # raw image
            resize = (imgs.size(2), imgs.size(3))
            transf = runner.cls_data_loader.img_transform
            loader = DataLoader(ImageFilelist(list(zip(paths, labels.cpu())), resize=resize, transform=transf),
                       batch_size=NUM_MKGRID_IMGS, shuffle=False, num_workers=1, pin_memory=True)
            imgs, _ = next(iter(loader))
            grid = torchvision.utils.make_grid(imgs.data[:NUM_MKGRID_IMGS], nrow=8, padding=0)
            mm.add_image('train', f'Raw-{attr}-{self._report_images_num}', grid)

        mm.add_scalar('train', 'learning_rate', x=iters, y=runner.optimizer.param_groups[0]['lr'])
        mm.send()

    def handle_validation(self, runner):
        iters = runner.runner_state['iters']
        epoch = runner.runner_state['epoch']

        mm = MessageMetric()
        # loss: train and val
        train_loss = list(runner.train_losses.avg.values())[0]
        val_loss = list(runner.val_losses.avg.values())[0]
        mm.add_scalar('train_val', 'loss', x=iters, y=[train_loss, val_loss])

        # weight, bias, grad
        if self._epoch != epoch and epoch < MAX_CONV2D_HIST:
            for key, module in self._model.named_modules():
                if not isinstance(module, torch.nn.Conv2d):
                    continue
                if module.weight is not None:
                    mm.add_histogram('train', 'conv2d_1_weight', module.weight.data, epoch)
                    mm.add_histogram('train', 'conv2d_1_weight.grad', module.weight.grad.data, epoch)
                if module.bias is not None:
                    mm.add_histogram('train', 'conv2d_1_bias', module.bias.data)
                    mm.add_histogram('train', 'conv2d_1_bias.grad', module.bias.grad.data, epoch)
                break

        # acc
        y = {
            'top1': runner.running_score.get_top1_acc()['out'],
            'top3': runner.running_score.get_top3_acc()['out'],
            'top5': runner.running_score.get_top5_acc()['out']
        }
        mm.add_scalar('train_val', 'acc', x=iters, y=y)
        mm.send()

        self._epoch = epoch

    def handle_evaluate(self, runner):
        pass


class ClsRunner(RunnerBase):
    def __init__(self, data_dir, model):
        super().__init__(data_dir, model)

    def handle_evaluate(self, runner, y_true, y_pred, files):
        mm = MessageMetric()
        # confusion matrix
        if runner.configer.get('data.num_classes') < 100:
            cm = confusion_matrix(y_true, y_pred)
            mm.add_matrix('evaluate', 'confusion_matrix', cm)

        # precision, recall, fscore
        P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        mm.add_text('evaluate', 'precision', P)
        mm.add_text('evaluate', 'recall', R)
        mm.add_text('evaluate', 'fscore', F)

        mm.send()


class DetRunner(RunnerBase):
    def __init__(self, data_dir, model):
        super().__init__(data_dir, model)


class RunnerStat(object):
    H = None

    @staticmethod
    def check(runner):
        if RunnerStat.H is None:
            data_dir = runner.configer.get('data', 'data_dir')
            task = runner.configer.get('task')
            if task == 'cls':
                RunnerStat.H = ClsRunner(data_dir, runner.cls_net)
            elif task == 'det':
                RunnerStat.H = DetRunner(data_dir, runner.det_net)
            else:
                raise NotImplementedError

    @staticmethod
    def train(runner, data_dict):
        RunnerStat.check(runner)
        RunnerStat.H.handle_train(runner, data_dict)

    @staticmethod
    def validation(runner):
        RunnerStat.check(runner)
        RunnerStat.H.handle_validation(runner)

    @staticmethod
    def evaluate(runner, *args, **kwargs):
        RunnerStat.check(runner)
        RunnerStat.H.handle_evaluate(runner, *args, **kwargs)
