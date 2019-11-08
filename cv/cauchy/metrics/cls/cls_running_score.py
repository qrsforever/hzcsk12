#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Image classification running score.

from __future__ import absolute_import, division, print_function

import torch

from cauchy.utils.tools.average_meter import AverageMeter


class ClsRunningScore(object):
    def __init__(self, configer):
        self.configer = configer
        self.top1_acc = AverageMeter()
        self.top3_acc = AverageMeter()
        self.top5_acc = AverageMeter()
        self.top1_pred = []
        self.top3_pred = []
        self.top5_pred = []

    def get_top1_acc(self):
        return self.top1_acc.avg

    def get_top3_acc(self):
        return self.top3_acc.avg

    def get_top5_acc(self):
        return self.top5_acc.avg

    def get_pred(self):
        top1_pred = torch.cat(tuple(self.top1_pred), dim=0).cpu().numpy()
        top3_pred = torch.cat(tuple(self.top3_pred), dim=0).cpu().numpy()
        top5_pred = torch.cat(tuple(self.top5_pred), dim=0).cpu().numpy()
        return (top1_pred, top3_pred, top5_pred)

    def update(self, output, target):
        """Computes the precision@k for the specified values of k"""
        topk = (1, 3, 5)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        _pred_1 = pred[0, :]
        _pred_3 = _pred_1.clone()
        _pred_5 = _pred_1.clone()
        for k, _pred in zip(topk, [_pred_1, _pred_3, _pred_5]):
            correct_k = correct[:k]
            correct_k_sum = correct_k.view(-1).float().sum(0, keepdim=False)
            correct_k = correct_k.sum(dim=0)
            _pred[correct_k == 1] = target[correct_k == 1]
            res.append(correct_k_sum / batch_size)
        self.top1_pred.append(_pred_1)
        self.top3_pred.append(_pred_3)
        self.top5_pred.append(_pred_5)

        self.top1_acc.update(res[0].item(), batch_size)
        self.top3_acc.update(res[1].item(), batch_size)
        self.top5_acc.update(res[2].item(), batch_size)

    def reset(self):
        self.top1_acc.reset()
        self.top3_acc.reset()
        self.top5_acc.reset()
