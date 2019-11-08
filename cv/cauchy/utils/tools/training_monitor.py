#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

import visdom
import json

from .machine_status_monitor import get_cpu_status, get_gpu_status
from ..helpers.visdom_helper import create_text_window


class TrainingMonitor:
    def __init__(self, configer):
        self.task_type = configer.get("task")
        self.training_stats = self._init_states(configer)
        self.vis = visdom.Visdom(port=configer.get("viz_port"))
        self.monitor_window = create_text_window(self.vis, "training_monitor")

    def _init_states(self, configer):
        """initialize monitor stats templates"""
        monitor_stats = {
            "cls": {
                "iters": 0,
                "lr": 0,
                "training_loss": 0,
                "training_speed": 0,
                "GPU": {},
                "CPU": {},
                "val_loss": 0,
                "acc": {},
            },
            "det": {
                "iters": 0,
                "lr": 0,
                "training_loss": 0,
                "training_speed": 0,
                "GPU": {},
                "CPU": {},
                "val_loss": 0,
                "ap_list": {},
                "mAP": {},
            },
            "seg": {
                "iters": 0,
                "lr": 0,
                "training_loss": 0,
                "training_speed": 0,
                "GPU": {},
                "CPU": {},
                "val_loss": 0,
                "mIOU": 0,
                "pixel_acc": 0,
            },
            "automl": {
                "iters": 0,
                "lr": 0,
                "training_loss": 0,
                "arch_training_loss": 0,
                "normal_cell": None,
                "reduce_cell": None,
                "training_speed": 0,
                "GPU": {},
                "CPU": {},
                "val_loss": 0,
                "acc": {},
            },
        }
        try:
            return monitor_stats[self.task_type]
        except KeyError:
            print("task type is not supported")
            raise

    def _update_training_state(
        self,
        iters,
        lr,
        training_loss,
        training_speed,
        arch_training_loss=None,
        normal_cell=None,
        reduce_cell=None,
    ):
        assert iters is not None, "params shouldn't be None"
        assert training_loss is not None, "params shouldn't be None"
        assert training_speed is not None, "params shouldn't be None"
        assert lr is not None, "params shouldn't be None"
        self.training_stats["iters"] = iters
        self.training_stats["lr"] = lr[0]
        self.training_stats["training_loss"] = training_loss
        self.training_stats["training_speed"] = training_speed
        self.training_stats["GPU"] = get_gpu_status()
        self.training_stats["CPU"] = get_cpu_status()
        if arch_training_loss:
            self.training_stats["arch_training_loss"] = arch_training_loss
        if normal_cell:
            self.training_stats["normal_cell"] = normal_cell
        if reduce_cell:
            self.training_stats["reduce_cell"] = reduce_cell

    def _update_cls_stats(
        self,
        iters=None,
        lr=None,
        training_loss=None,
        training_speed=None,
        acc=None,
        val_loss=None,
        phase=None,
        *args,
        **kwargs
    ):
        assert phase in ("train", "val"), "Phase must be train or val"
        if phase == "train":
            self._update_training_state(
                iters, lr, training_loss, training_speed
            )
        elif phase == "val":
            assert acc is not None, "params shouldn't be None"
            assert val_loss is not None, "params shouldn't be None"
            self.training_stats["acc"] = acc
            self.training_stats["val_loss"] = val_loss

    def _update_det_stats(
        self,
        iters=None,
        lr=None,
        training_loss=None,
        training_speed=None,
        ap_list=None,
        mAP=None,
        val_loss=None,
        phase=None,
        *args,
        **kwargs
    ):
        assert phase in ("train", "val"), "Phase must be in train or val"
        if phase == "train":
            self._update_training_state(
                iters, lr, training_loss, training_speed
            )
        elif phase == "val":
            assert ap_list is not None, "params shouldn't be None"
            assert val_loss is not None, "params shouldn't be None"
            assert mAP is not None, "params shouldn't be None"
            self.training_stats["ap_list"] = ",".join(
                [str(ap) for ap in ap_list]
            )
            self.training_stats["mAP"] = mAP
            self.training_stats["val_loss"] = val_loss

    def _update_seg_stats(
        self,
        iters=None,
        lr=None,
        training_loss=None,
        training_speed=None,
        val_loss=None,
        miou=None,
        pixel_acc=None,
        phase=None,
        *args,
        **kwargs
    ):
        assert phase in ("train", "val"), "Phase must be in train or val"
        if phase == "train":
            self._update_training_state(
                iters, lr, training_loss, training_speed
            )
        elif phase == "val":
            assert miou is not None, "params shouldn't be None"
            assert val_loss is not None, "params shouldn't be None"
            assert pixel_acc is not None, "params shouldn't be None"
            self.training_stats["val_loss"] = val_loss
            self.training_stats["mIOU"] = miou
            self.training_stats["pixel_acc"] = pixel_acc

    def _update_automl_stats(
        self,
        iters=None,
        lr=None,
        training_loss=None,
        arch_training_loss=None,
        normal_cell=None,
        reduce_cell=None,
        training_speed=None,
        acc=None,
        val_loss=None,
        phase=None,
        *args,
        **kwargs
    ):

        assert phase in (
            "train",
            "val",
            "finetune",
        ), "Phase must be train/val/finetune"
        if phase == "train":
            assert arch_training_loss is not None, "params shouldn't be None"
        if phase in ["train", "finetune"]:
            self._update_training_state(
                iters,
                lr,
                training_loss,
                training_speed,
                arch_training_loss,
                normal_cell,
                reduce_cell,
            )
        elif phase == "val":
            assert acc is not None, "params shouldn't be None"
            assert val_loss is not None, "params shouldn't be None"
            self.training_stats["acc"] = acc
            self.training_stats["val_loss"] = val_loss

    def update(self, *args, **kwargs):
        if self.task_type == "cls":
            self._update_cls_stats(*args, **kwargs)
        elif self.task_type == "det":
            self._update_det_stats(*args, **kwargs)
        elif self.task_type == "seg":
            self._update_seg_stats(*args, **kwargs)
        elif self.task_type == "automl":
            self._update_automl_stats(*args, **kwargs)

    def flush(self,):
        """send out training stats"""
        self.vis.text(
            text=json.dumps(self.training_stats), win=self.monitor_window
        )

    def stop_signal(self):
        self.vis.text(
            text=json.dumps({"status": "finihsed"}), win=self.monitor_window
        )
