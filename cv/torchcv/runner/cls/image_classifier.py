#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Image Classifier.


import time
import torch

from lib.runner.runner_helper import RunnerHelper
from lib.runner.trainer import Trainer
from lib.tools.util.average_meter import AverageMeter, DictAverageMeter
from lib.tools.util.logger import Logger as Log
from metric.cls.cls_running_score import ClsRunningScore
from model.cls.model_manager import ModelManager
from data.cls.data_loader import DataLoader

# QRS: add
from k12ai.runner.stat import RunnerStat
from model.cls.loss.loss import BASE_LOSS_DICT


class ImageClassifier(object):
    """
      The class for the training phase of Image classification.
    """
    def __init__(self, configer):
        self.configer = configer
        self.runner_state = dict()

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = DictAverageMeter()
        self.val_losses = DictAverageMeter()
        self.cls_model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.running_score = ClsRunningScore(configer)

        self.cls_net = self.cls_model_manager.get_cls_model()
        self.solver_dict = self.configer.get('solver')
        self.cls_net = RunnerHelper.load_net(self, self.cls_net)
        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.solver_dict)
        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()
        self.loss = self.cls_model_manager.get_cls_loss()
        # QRS: add
        self.valid_loss_dict = configer.get('loss.loss_weights', configer.get('loss.loss_type'))
        if self.configer.get('network', 'resume_continue'):
            if 'epoch' in self.runner_state and self.runner_state['epoch'] > 0:
                self.runner_state['epoch'] -= 1
                self.runner_state['iters'] = self.runner_state['epoch'] * len(self.train_loader)

    def _get_parameters(self):
        if self.solver_dict.get('optim.wdall', default=True):
            lr_1 = []
            lr_2 = []
            params_dict = dict(self.cls_net.named_parameters())
            for key, value in params_dict.items():
                if value.requires_grad:
                    if 'backbone' in key:
                        if self.configer.get('solver.lr.bb_lr_scale') == 0.0:
                            value.requires_grad = False
                        else:
                            lr_1.append(value)
                    else:
                        lr_2.append(value)

            params = [
                {'params': lr_1, 'lr': self.solver_dict['lr']['base_lr'] * self.configer.get('solver.lr.bb_lr_scale')},
                {'params': lr_2, 'lr': self.solver_dict['lr']['base_lr']}]
        else:
            no_decay_list = []
            decay_list = []
            no_decay_name = []
            decay_name = []
            for m in self.cls_net.modules():
                if (hasattr(m, 'groups') and m.groups > 1) or isinstance(m, torch.nn.BatchNorm2d) \
                        or m.__class__.__name__ == 'GL':
                    no_decay_list += m.parameters(recurse=False)
                    for name, p in m.named_parameters(recurse=False):
                        no_decay_name.append(m.__class__.__name__ + name)
                else:
                    for name, p in m.named_parameters(recurse=False):
                        if 'bias' in name:
                            no_decay_list.append(p)
                            no_decay_name.append(m.__class__.__name__ + name)
                        else:
                            decay_list.append(p)
                            decay_name.append(m.__class__.__name__ + name)
            Log.info('no decay list = {}'.format(no_decay_name))
            Log.info('decay list = {}'.format(decay_name))
            params = [{'params': no_decay_list, 'weight_decay': 0}, {'params': decay_list}]

        return params

    # QRS: add
    def get_loss_params(self, out, targets):
        loss_dict = {}
        if 'ce_loss' in self.valid_loss_dict:
            loss_dict['ce_loss'] = dict(
                params=[out, targets],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['ce_loss']])
            )
        return loss_dict

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.cls_net.train()
        # Adjust the learning rate after every epoch.
        self.runner_state['epoch'] += 1
        for i, data_dict in enumerate(self.train_loader):
            start_time = time.time()
            self.data_time.update(time.time() - start_time)
            data_dict = RunnerHelper.to_device(self, data_dict)
            # Forward pass.

            # QRS: fuck
            out = self.cls_net(data_dict['img'])
            loss_dict = self.loss(self.get_loss_params(out, data_dict['label']))

            # Compute the loss of the train batch & backward.

            loss = loss_dict['loss']
            self.train_losses.update({key: loss.item() for key, loss in loss_dict.items()}, data_dict['img'].size(0))
            self.optimizer.zero_grad()
            loss.backward()
            if self.configer.get('network', 'clip_grad', default=False):
                RunnerHelper.clip_grad(self.cls_net, 10.)

            self.optimizer.step()

            # QRS
            Trainer.update(self, warm_list=(0, 1),
                           warm_lr_list=(self.solver_dict['lr']['base_lr']*self.configer.get('solver.lr.bb_lr_scale'),
                                         self.solver_dict['lr']['base_lr']),
                           solver_dict=self.solver_dict)

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            self.runner_state['iters'] += 1

            # Print the log info & reset the states.
            if self.runner_state['iters'] % self.solver_dict['display_iter'] == 0:
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {4}\tLoss = {3}\n'.format(
                             self.runner_state['epoch'], self.runner_state['iters'],
                             self.solver_dict['display_iter'], self.train_losses.info(),
                             RunnerHelper.get_lr(self.optimizer), batch_time=self.batch_time,
                             data_time=self.data_time))
                # print('%03d' % self.runner_state['epoch'],
                #         round(torch.cuda.max_memory_allocated()/2**20, 2),
                #         round(torch.cuda.max_memory_reserved()/2**20, 2),
                #         round(torch.cuda.memory_allocated()/2**20, 2),
                #         round(torch.cuda.memory_reserved()/2**20, 2))

                # QRS: add
                RunnerStat.train(self, data_dict)

                self.batch_time.reset()
                self.data_time.reset()

            del data_dict, loss, loss_dict

            if self.solver_dict['lr']['metric'] == 'iters' and self.runner_state['iters'] == self.solver_dict['max_iters']:
                break

            if self.runner_state['iters'] % self.solver_dict['save_iters'] == 0 and self.configer.get('local_rank') == 0:
                RunnerHelper.save_net(self, self.cls_net)

            # Check to val the current model.
            if self.runner_state['iters'] % self.solver_dict['test_interval'] == 0:
                # torch.cuda.empty_cache()
                self.val(False)
                self.train_losses.reset()

    def val(self, save=True):
        """
          Validation function during the train phase.
        """
        # QRS:
        if save:
            RunnerHelper.save_net(self, self.cls_net)
        if len(self.val_loader) == 0:
            return
        self.cls_net.eval()
        with torch.no_grad():
            for j, data_dict in enumerate(self.val_loader):
                start_time = time.time()
                # Forward pass.
                data_dict = RunnerHelper.to_device(self, data_dict)
                # QRS: mod
                out = self.cls_net(data_dict['img'])
                loss_dict = self.loss(self.get_loss_params(out, data_dict['label']))
                # out_dict, label_dict, _ = RunnerHelper.gather(self, out)
                self.running_score.update({'out': out}, {'out': data_dict['label']})
                self.val_losses.update({key: loss.item() for key, loss in loss_dict.items()}, data_dict['img'].size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)

            RunnerStat.validation(self)

            # Print the log info & reset the states.
            Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))
            Log.info('TestLoss = {}'.format(self.val_losses.info()))
            Log.info('Top1 ACC = {}'.format(RunnerHelper.dist_avg(self, self.running_score.get_top1_acc())))
            Log.info('Top3 ACC = {}'.format(RunnerHelper.dist_avg(self, self.running_score.get_top3_acc())))
            Log.info('Top5 ACC = {}'.format(RunnerHelper.dist_avg(self, self.running_score.get_top5_acc())))
            self.batch_time.reset()
            self.val_losses.reset()
            self.running_score.reset()
            self.cls_net.train()


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
