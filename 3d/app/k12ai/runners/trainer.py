#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-21 13:16

import time
import torch
from torch.optim.lr_scheduler import ( # noqa
    StepLR,
    MultiStepLR,
    ReduceLROnPlateau
)

from k12ai.common.log_message import MessageMetric


class Trainer(object):

    def __init__(self,
            train_dataloader, valid_dataloader,
            model, criterion, optimizer, scheduler,
            num_epoch, metrics, cache_dir):

        self.iters = 0
        self.epoch = 0
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epoch = num_epoch
        self.cache_dir = cache_dir
        self.metrics = metrics
        self.log_freq = metrics.get('log_freq', default=100)

    def train(self):
        raise NotImplementedError


class DepthPredictTrainer(Trainer):

    def __init__(self,
            train_dataloader, valid_dataloader,
            model, criterion, optimizer, scheduler,
            num_epoch, metrics, cache_dir):
        super().__init__(train_dataloader, valid_dataloader,
                model, criterion, optimizer, scheduler,
                num_epoch, metrics, cache_dir)

        from k12ai.metrics.fcrn import AverageMeter
        self.average_meter = AverageMeter()

    def _log_metrics(self, phase, avg):
        mm = MessageMetric()
        if self.metrics.get(f'{phase}.loss', default=False):
            mm.add_scalar('phase', 'loss', x=self.iters, y=avg.loss)
        if self.metrics.get(f'{phase}.absrel', default=False) :
            mm.add_scalar('phase', 'absrel', x=self.iters, y=avg.absrel)
        if self.metrics.get(f'{phase}.mse', default=False) :
            mm.add_scalar('phase', 'mse', x=self.iters, y=avg.mse)
        if self.metrics.get(f'{phase}.rmse', default=False):
            mm.add_scalar('phase', 'rmse', x=self.iters, y=avg.rmse)
        if self.metrics.get(f'{phase}.mae', default=False):
            mm.add_scalar('phase', 'mae', x=self.iters, y=avg.mae)
        if self.metrics.get(f'{phase}.irmse', default=False):
            mm.add_scalar('phase', 'irmse', x=self.iters, y=avg.irmse)
        if self.metrics.get(f'{phase}.imae', default=False):
            mm.add_scalar('phase', 'imae', x=self.iters, y=avg.imae)
        if self.metrics.get(f'{phase}.delta1', default=False):
            mm.add_scalar('phase', 'delta1', x=self.iters, y=avg.delta1)
        if self.metrics.get(f'{phase}.delta2', default=False):
            mm.add_scalar('phase', 'delta2', x=self.iters, y=avg.delta2)
        if self.metrics.get(f'{phase}.delta3', default=False):
            mm.add_scalar('phase', 'delta3', x=self.iters, y=avg.delta3)
        if self.metrics.get(f'{phase}.speed', default=False):
            mm.add_scalar('phase', 'speed', x=self.iters, y=1.0 / avg.batchtime)
        mm.send()

    def _train_epoch(self, epoch):
        self.average_meter.reset()
        self.model.train()
        start_time = time.time()
        for data, target in self.train_dataloader:
            self.iters += 1
            data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            criterion = self.criterion(output, target)
            criterion.backward()
            self.optimizer.step()
            self.average_meter.update(output, target, criterion.item(), time.time() - start_time)
            if self.iters % self.log_freq == 0:
                self._log_metrics('train', self.average_meter.average())
        return self.average_meter.average()

    def _valid_epoch(self, epoch):
        self.average_meter.reset()
        self.model.eval()
        with torch.no_grad():
            start_time = time.time()
            for data, target in self.valid_dataloader:
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                criterion = self.criterion(output, target)
                self.average_meter.update(output, target, criterion.item(), time.time() - start_time)
        return self.average_meter.average()

    def train(self):
        for epoch in range(0, self.num_epoch):
            self.epoch += 1
            avg = self._train_epoch(epoch)
            self._log_metrics('train', avg)
            avg = self._valid_epoch(epoch)
            if isinstance(self.scheduler, (ReduceLROnPlateau,)):
                self.scheduler.step(avg.absrel)
            else:
                self.scheduler.step()

    def evaluate(self):
        self.average_meter.reset()
        self.model.eval()
        with torch.no_grad():
            start_time = time.time()
            for data, target in self.valid_dataloader:
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                criterion = self.criterion(output, target)
                self.average_meter.update(output, target, criterion.item(), time.time() - start_time)
        return self.average_meter.average()
