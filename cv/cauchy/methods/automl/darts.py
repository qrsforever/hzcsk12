#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

from cauchy.datasets.automl.data_loader import Cifar10Loader as DataLoader
from cauchy.methods.tools.runner_helper import RunnerHelper
from cauchy.methods.tools.trainer import Trainer
from cauchy.metrics.cls.cls_running_score import ClsRunningScore
from cauchy.models.automl.darts import utils
from cauchy.models.automl.darts.arch import Arch
from cauchy.models.automl.darts.model_search import Network as SearchNetwork
from cauchy.models.automl.darts.model import NetworkCIFAR as FinetuneNetwork
from cauchy.utils.tools.average_meter import AverageMeter
from cauchy.utils.tools.logger import Logger as Log
from cauchy.utils.tools.training_monitor import TrainingMonitor
from cauchy.utils.helpers.file_helper import FileHelper
from cauchy.utils.tools.load_custom_model import load_genotype


class Darts(object):
    """The class for the training phase of Image classification.
  In order to train darts, following params should be specified:
  * `seed`: seed for torch and numpy
  * `genotype`: orignal architecture
  * `init_ch`: number of init channels
  * `layers_num` : total num of layers
  """

    def __init__(self, configer):
        cudnn.benchmark = True
        self.configer = configer
        self.phase = self.configer.get("phase")
        assert self.phase in ["train", "finetune"], "invalid phase"

        # set up numpy and torch env
        np.random.seed(self.configer.get("network", "seed"))
        # torch.cuda.set_device(self.configer.get('gpu'))
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.manual_seed(self.configer.get("network", "seed"))

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.data_loader = DataLoader(configer)
        self.cls_running_score = ClsRunningScore(configer)
        self.device = torch.device("cuda:0")
        self.training_monitor = TrainingMonitor(configer)

        self.runner_state = dict()

        self._init_model()

    def _init_model(self):
        # initialize criterion
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # copy genotype file to checkpoint dir
        self.ckpt_dir = os.path.join(
            self.configer.get("project_dir"),
            self.configer.get("network", "checkpoints_root"),
            self.configer.get("network", "checkpoints_dir"),
        )
        FileHelper.make_dirs(self.ckpt_dir)
        if self.phase == "train":
            # write initial content for current traning
            utils.write_genotype(self.ckpt_dir)

            self.model = SearchNetwork(
                self.configer.get("network", "init_ch"),
                self.configer.get("data", "num_classes"),
                self.configer.get("network", "layers_num"),
                self.criterion,
            ).to(self.device)

            # intialize model architecture
            self.arch = Arch(self.model, self.configer)

            # set up data loader
            self.train_queue, self.val_queue = self.data_loader.get_trainloader(
                search=True
            )
            self.val_loader = self.data_loader.get_valloader()

            # setup optimizer and lr training scheduler
            self.optimizer, self.scheduler = Trainer.init(
                self._get_parameters(), self.configer.get("solver")
            )
        else:
            # load searched genotype
            genotype = load_genotype(
                os.path.join(self.ckpt_dir, "genotypes.py"),
                self.configer.get("network", "search_epoch"),
            )

            # generate network to finetune
            self.model = FinetuneNetwork(
                self.configer.get("network", "init_ch"),
                self.configer.get("data", "num_classes"),
                self.configer.get("network", "layers_num"),
                self.configer.get("network", "auxiliary"),
                genotype,
            )
            self.model.to(self.device)

            # set up dataloader
            self.train_loader = self.data_loader.get_trainloader(search=False)
            self.val_queue = self.data_loader.get_valloader()

            # set up optimizer and lr scheduler
            self.optimizer, self.scheduler = Trainer.init(
                self._get_parameters(), self.configer.get("solver")
            )

    def _get_parameters(self):
        return self.model.parameters()

    def train(self):
        # search architecture for epochs times, during each each, network params
        # and architecture of networks are updated alternatively
        self.model.train()
        cur_lr = RunnerHelper.get_lr(self.optimizer)[0]
        Log.info("epoch {} lr {}".format(self.runner_state["epoch"], cur_lr))

        genotype = self.model.genotype()
        Log.info("genotype = {}".format(genotype))
        # start training

        valid_iter = iter(self.val_queue)

        self.runner_state["epoch"] += 1
        start_time = time.time()

        for step, (x, target) in enumerate(self.train_queue):
            cur_genotype = self.model.genotype()
            Trainer.update(self, solver_dict=self.configer.get("solver"))
            batchsz = x.size(0)

            # [b, 3, 32, 32], [40]
            x = RunnerHelper.to_device(self, x)
            # x = x.to(self.device)
            if self.configer.get("gpu"):
                target = target.cuda(non_blocking=True)
            x_search, target_search = next(valid_iter)  # [b, 3, 32, 32], [b]
            x_search = RunnerHelper.to_device(self, x_search)
            # x_search = x_search.to(self.device)

            target_search = target_search.cuda(non_blocking=True)

            # 1. update alpha
            arch_loss = self.arch.step(
                x,
                target,
                x_search,
                target_search,
                cur_lr,
                self.optimizer,
                unrolled=self.configer.get("network", "unrolled"),
            )

            logits = self.model(x)
            loss = self.criterion(logits, target)
            training_loss = loss.detach().cpu().numpy()

            # update info
            self.train_losses.update(loss.item(), batchsz)
            self.cls_running_score.update(logits, target)

            # 2. update weight
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.configer.get("network", "grad_clip"),
            )
            self.optimizer.step()
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.runner_state["iters"] += 1

            # update training monitor
            self.training_monitor.update(
                iters=self.runner_state["iters"],
                lr=RunnerHelper.get_lr(self.optimizer),
                training_loss=float(training_loss),
                training_speed=1 / self.batch_time.val,
                arch_training_loss=float(arch_loss),
                normal_cell=cur_genotype.normal,
                reduce_cell=cur_genotype.reduce,
                phase="train",
            )

            if step % self.configer.get("solver", "display_iter") == 0:
                Log.info(
                    "Step:{} loss:{:.2f} acc1:{:.2f} acc5:{:.2f}".format(
                        self.runner_state["iters"],
                        self.train_losses.avg,
                        self.cls_running_score.get_top1_acc(),
                        self.cls_running_score.get_top5_acc(),
                    )
                )
                self.cls_running_score.reset()
                self.train_losses.reset()

            # validate model
            if (
                self.runner_state["iters"]
                % self.configer.get("solver", "test_interval")
                == 0
            ):
                acc_val, val_loss = self.val()
                self.training_monitor.update(
                    acc=acc_val, val_loss=val_loss, phase="val"
                )

            self.training_monitor.flush()

        # update genotype
        new_genotype = "DARTS_EPOCH_{} = {}".format(
            self.runner_state["epoch"], self.model.genotype()
        )

        utils.write_genotype(self.ckpt_dir, new_genotype)
        Log.info(
            "update genotype for epoch {}".format(self.runner_state["epoch"])
        )

    def val(self):
        self.model.eval()

        with torch.no_grad():
            for step, (x, target) in enumerate(self.val_queue):

                x, target = x.to(self.device), target.cuda(non_blocking=True)
                batchsz = x.size(0)

                if self.configer.get("phase") == "train":
                    logits = self.model(x)
                else:
                    logits, _ = self.model(x)
                loss = self.criterion(logits, target)

                # update info
                self.val_losses.update(loss.item(), batchsz)
                self.cls_running_score.update(logits, target)

        Log.info(
            "Val stats for epoch {}: {}, {},{}".format(
                self.runner_state["epoch"],
                self.val_losses.avg,
                self.cls_running_score.get_top1_acc(),
                self.cls_running_score.get_top5_acc(),
            )
        )

        acc_val = {
            "top1": self.cls_running_score.get_top1_acc(),
            "top3": self.cls_running_score.get_top3_acc(),
            "top5": self.cls_running_score.get_top5_acc(),
        }
        val_loss = self.val_losses.avg

        self.val_losses.reset()
        self.cls_running_score.reset()

        return acc_val, val_loss

    def finetune(self):
        self.model.drop_path_prob = (
            self.configer.get("network", "drop_path_prob")
            * self.runner_state["epoch"]
            / self.configer.get("solver", "max_epoch")
        )

        self.model.train()
        self.runner_state["epoch"] += 1
        start_time = time.time()

        for step, (x, target) in enumerate(self.train_loader):
            x = x.to(self.device)
            target = target.cuda(non_blocking=True)

            batchsz = x.size(0)

            self.optimizer.zero_grad()

            logits, logits_aux = self.model(x)
            loss = self.criterion(logits, target)

            # update info
            self.train_losses.update(loss.item(), batchsz)
            self.cls_running_score.update(logits, target)

            if self.configer.get("network", "auxiliary"):
                loss_aux = self.criterion(logits_aux, target)
                loss += (
                    self.configer.get("network", "auxiliary_weight") * loss_aux
                )
            loss_val = loss.detach().cpu().numpy()
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.configer.get("network", "grad_clip"),
            )
            self.optimizer.step()

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.runner_state["iters"] += 1
            # update training monitor
            self.training_monitor.update(
                iters=self.runner_state["iters"],
                lr=RunnerHelper.get_lr(self.optimizer),
                training_loss=float(loss_val),
                training_speed=1 / self.batch_time.val,
                phase="finetune",
            )

            if step % self.configer.get("solver", "display_iter") == 0:
                Log.info(
                    "Step:{} loss:{:.2f} acc1:{:.2f} acc5:{:.2f}".format(
                        self.runner_state["iters"],
                        self.train_losses.avg,
                        self.cls_running_score.get_top1_acc(),
                        self.cls_running_score.get_top5_acc(),
                    )
                )
                self.cls_running_score.reset()
                self.train_losses.reset()

            # validate model
            if (
                self.runner_state["epoch"]
                % self.configer.get("solver", "test_interval")
                == 0
            ):
                acc_val, val_loss = self.val()
                print(acc_val)
                self.training_monitor.update(
                    acc=acc_val, val_loss=val_loss, phase="val"
                )

            self.training_monitor.flush()
