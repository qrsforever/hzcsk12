#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file fcrn.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-21 12:39

import torch
import math


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


class Result(object):
    def __init__(self):
        self.loss = .0
        self.irmse, self.imae = .0, .0
        self.mse, self.rmse, self.mae = .0, .0, .0
        self.absrel, self.lg10 = .0, .0
        self.delta1, self.delta2, self.delta3 = .0, .0, .0
        self.batchtime = 0

    def __str__(self):
        ss  = ('-' * 50) # noqa
        ss += f'\t\n     loss:  {self.loss}\n'
        ss += f'\t   absrel:  {self.absrel}\n'
        ss += f'\t      mse:  {self.mse}\n'
        ss += f'\t     rmse:  {self.rmse}\n'
        ss += f'\t      mae:  {self.mae}\n'
        ss += f'\t    irmse:  {self.irmse}\n'
        ss += f'\t     imae:  {self.imae}\n'
        ss += f'\t     lg10:  {self.lg10}\n'
        ss += f'\t   delta1:  {self.delta1}\n'
        ss += f'\t   delta2:  {self.delta2}\n'
        ss += f'\t   delta3:  {self.delta3}\n'
        ss += f'\tbatchtime:  {self.batchtime}\n'
        return ss

    def update(self, loss, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, batchtime):
        self.loss = loss
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.batchtime = batchtime

    def evaluate(self, output, target, loss):
        valid_mask = target > 0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.loss = loss
        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.batchtime = 0

        inv_output = 1.0 / output
        inv_target = 1.0 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_loss = 0.0
        self.sum_irmse, self.sum_imae = 0.0, 0.0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0.0, 0.0, 0.0
        self.sum_absrel, self.sum_lg10 = 0.0, 0.0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0.0, 0.0, 0.0
        self.sum_batchtime = 0.0

    def update(self, output, target, loss, batchtime, n=1):
        result = Result()
        result.evaluate(output, target, loss)

        self.count += n

        self.sum_loss = n * result.loss
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_batchtime += n * batchtime

    def average(self):
        avg = Result()
        avg.update(
            self.sum_loss / self.count,
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_batchtime / self.count)
        return avg
