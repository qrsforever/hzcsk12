#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Class Definition for Yolo v3.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np

from cauchy.datasets.det.data_loader import DataLoader
from cauchy.methods.det.yolov3_test import YOLOv3Test
from cauchy.methods.tools.runner_helper import RunnerHelper
from cauchy.methods.tools.trainer import Trainer
from cauchy.models.det.model_manager import ModelManager
from cauchy.models.det.layers.yolo_detection_layer import YOLODetectionLayer
from cauchy.models.det.layers.yolo_target_generator import YOLOTargetGenerator
from cauchy.utils.tools.average_meter import AverageMeter
from cauchy.utils.tools.logger import Logger as Log
from cauchy.metrics.det.det_running_score import DetRunningScore
from cauchy.utils.visualizer.det_visualizer import DetVisualizer
from cauchy.utils.tools.training_monitor import TrainingMonitor


class YOLOv3(object):
    """
      The class for YOLO v3. Include train, val, test & predict.
    """

    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.det_visualizer = DetVisualizer(configer)
        self.det_model_manager = ModelManager(configer)
        self.det_data_loader = DataLoader(configer)
        self.yolo_detection_layer = YOLODetectionLayer(configer)
        self.yolo_target_generator = YOLOTargetGenerator(configer)
        self.det_running_score = DetRunningScore(configer)
        self.training_monitor = TrainingMonitor(configer)

        self.det_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.runner_state = dict()

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = RunnerHelper.load_net(self, self.det_net)

        self.optimizer, self.scheduler = Trainer.init(
            self._get_parameters(), self.configer.get("solver")
        )

        self.train_loader = self.det_data_loader.get_trainloader()
        self.val_loader = self.det_data_loader.get_valloader()

        self.det_loss = self.det_model_manager.get_det_loss()

    def _get_parameters(self):
        lr_1 = []
        lr_10 = []
        params_dict = dict(self.det_net.named_parameters())
        for key, value in params_dict.items():
            if "backbone" not in key:
                lr_10.append(value)
            else:
                lr_1.append(value)

        params = [
            {
                "params": lr_1,
                "lr": self.configer.get("solver", "lr")["base_lr"],
            },
            {
                "params": lr_10,
                "lr": self.configer.get("solver", "lr")["base_lr"] * 10.0,
            },
        ]

        return params

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.det_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.runner_state["epoch"] += 1

        # data_tuple: (inputs, heatmap, maskmap, vecmap)
        for i, data_dict in enumerate(self.train_loader):
            Trainer.update(
                self,
                backbone_list=(0,),
                solver_dict=self.configer.get("solver"),
            )
            inputs = data_dict["img"]
            batch_gt_bboxes = data_dict["bboxes"]
            batch_gt_labels = data_dict["labels"]
            input_size = [inputs.size(3), inputs.size(2)]

            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs = RunnerHelper.to_device(self, inputs)

            # Forward pass.
            feat_list, predictions, _ = self.det_net(inputs)

            targets, objmask, noobjmask = self.yolo_target_generator(
                feat_list, batch_gt_bboxes, batch_gt_labels, input_size
            )
            targets, objmask, noobjmask = RunnerHelper.to_device(
                self, targets, objmask, noobjmask
            )
            # Compute the loss of the train batch & backward.
            loss = self.det_loss(predictions, targets, objmask, noobjmask)

            self.train_losses.update(loss.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.runner_state["iters"] += 1

            self.training_monitor.update(
                iters=self.runner_state["iters"],
                training_loss=float(loss.detach().cpu().numpy()),
                lr=RunnerHelper.get_lr(self.optimizer),
                training_speed=1 / self.batch_time.val,
                phase="train",
            )

            # Print the log info & reset the states.
            if (
                self.runner_state["iters"]
                % self.configer.get("solver", "display_iter")
                == 0
            ):
                Log.info(
                    "Train Epoch: {0}\tTrain Iteration: {1}\t"
                    "Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t"
                    "Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n"
                    "Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n".format(
                        self.runner_state["epoch"],
                        self.runner_state["iters"],
                        self.configer.get("solver", "display_iter"),
                        RunnerHelper.get_lr(self.optimizer),
                        batch_time=self.batch_time,
                        data_time=self.data_time,
                        loss=self.train_losses,
                    )
                )
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            if self.configer.get("solver", "lr")[
                "metric"
            ] == "iters" and self.runner_state["iters"] == self.configer.get(
                "solver", "max_iters"
            ):
                break

            # Check to val the current model.
            if (
                self.runner_state["iters"]
                % self.configer.get("solver", "test_interval")
                == 0
            ):
                ap_list, mAP, val_loss_avg = self.val()
                self.training_monitor.update(
                    ap_list=ap_list, mAP=mAP, val_loss=val_loss_avg, phase="val"
                )
            self.training_monitor.flush()

    def val(self):
        """
          Validation function during the train phase.
        """
        self.det_net.eval()
        start_time = time.time()
        with torch.no_grad():
            for i, data_dict in enumerate(self.val_loader):
                inputs = data_dict["img"]
                batch_gt_bboxes = data_dict["bboxes"]
                batch_gt_labels = data_dict["labels"]
                input_size = [inputs.size(3), inputs.size(2)]
                # Forward pass.
                inputs = RunnerHelper.to_device(self, inputs)
                feat_list, predictions, detections = self.det_net(inputs)

                targets, objmask, noobjmask = self.yolo_target_generator(
                    feat_list, batch_gt_bboxes, batch_gt_labels, input_size
                )
                targets, objmask, noobjmask = RunnerHelper.to_device(
                    self, targets, objmask, noobjmask
                )

                # Compute the loss of the val batch.
                loss = self.det_loss(predictions, targets, objmask, noobjmask)
                self.val_losses.update(loss.item(), inputs.size(0))

                batch_detections = YOLOv3Test.decode(
                    detections, self.configer, input_size
                )
                batch_pred_bboxes = self.__get_object_list(
                    batch_detections, input_size
                )

                self.det_running_score.update(
                    batch_pred_bboxes, batch_gt_bboxes, batch_gt_labels
                )

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            RunnerHelper.save_net(
                self, self.det_net, iters=self.runner_state["iters"]
            )
            # Print the log info & res t the states.
            Log.info(
                "Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t"
                "Loss {loss.avg:.8f}\n".format(
                    batch_time=self.batch_time, loss=self.val_losses
                )
            )
            Log.info("Val mAP: {}".format(self.det_running_score.get_mAP()))
            (ap_list, mAP) = self.det_running_score.get_mAP()
            val_loss_avg = self.val_losses.avg
            self.det_running_score.reset()
            self.batch_time.reset()
            self.val_losses.reset()

            self.det_net.train()
            return ap_list, mAP, val_loss_avg

    def __get_object_list(self, batch_detections, input_size):
        batch_pred_bboxes = list()
        for idx, detections in enumerate(batch_detections):
            detections = detections.detach().cpu().numpy()
            if detections is not None:
                detections[:, 4] = np.around(detections[:, 4], decimals=2)
                batch_pred_bboxes.append(
                    detections[:, [0, 1, 2, 3, 6, 4]].tolist()
                )

        return batch_pred_bboxes


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
