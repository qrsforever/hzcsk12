#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-21 13:16


class Trainer(object):

    def train(self):
        raise NotImplementedError


class DepthPredictTrainer(Trainer):
    def __init__(
            self,
            train_dataloader, valid_dataloader,
            model, criterion, optimizer, scheduler,
            num_epoch, cache_dir):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epoch = num_epoch
        self.cache_dir = cache_dir

    def _train_epoch(self, epoch):
        self.model.train()
        for data, target in self.train_dataloader:
            data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            criterion = self.criterion(output, target)
            criterion.backward()
            self.optimizer.step()
            print(criterion.item())

    def _valid_epoch(self, epoch):
        self.model.eval()
        pass

    def train(self):
        for epoch in range(0, self.num_epoch):                                                                                                                        
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
            self.scheduler.step()
