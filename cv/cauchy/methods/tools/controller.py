#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Some methods used by main methods.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cauchy.utils.helpers.file_helper import FileHelper
from cauchy.utils.tools.logger import Logger as Log


class Controller(object):
    @staticmethod
    def init(runner):
        runner.runner_state["iters"] = 0
        runner.runner_state["last_iters"] = 0
        runner.runner_state["epoch"] = 0
        runner.runner_state["last_epoch"] = 0
        runner.runner_state["performance"] = 0
        runner.runner_state["val_loss"] = 0
        runner.runner_state["max_performance"] = 0
        runner.runner_state["min_val_loss"] = 0

        if runner.configer.get("phase") == "train":
            assert (
                len(runner.configer.get("gpu")) > 1
                or runner.configer.get("network", "bn_type") == "torchbn"
            )

        Log.info(
            "BN Type is {}.".format(runner.configer.get("network", "bn_type"))
        )

    @staticmethod
    def train(runner):
        Log.info("Training start...")
        if runner.configer.get(
            "network", "resume"
        ) is not None and runner.configer.get("network", "resume_val"):
            runner.val()

        if runner.configer.get("solver", "lr")["metric"] == "epoch":
            while runner.runner_state["epoch"] < runner.configer.get(
                "solver", "max_epoch"
            ):
                runner.train()
                if runner.runner_state["epoch"] == runner.configer.get(
                    "solver", "max_epoch"
                ):
                    runner.val()
                    break
        else:
            while runner.runner_state["iters"] < runner.configer.get(
                "solver", "max_iters"
            ):
                runner.train()
                if runner.runner_state["iters"] == runner.configer.get(
                    "solver", "max_iters"
                ):
                    runner.val()
                    break

        Log.info("Training end...")

    @staticmethod
    def debug(runner):
        Log.info("Debugging start..")
        base_dir = os.path.join(
            runner.configer.get("project_dir"),
            "out/vis",
            runner.configer.get("task"),
            runner.configer.get("network", "model_name"),
        )

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        runner.debug(base_dir)
        Log.info("Debugging end...")

    @staticmethod
    def test(runner):
        Log.info("Testing start...")
        test_img = runner.configer.get("test", "test_img")
        test_dir = runner.configer.get("test", "test_dir")
        if test_img is None and test_dir is None:
            Log.error("test_img & test_dir not exists.")
            raise RuntimeError("test_img & test_dir not exists.")

        if test_img is not None and test_dir is not None:
            Log.error("Either test_img or test_dir.")
            raise RuntimeError("test_img & test_dir not exists.")

        try:
            if test_img is not None:
                return runner.test_img(test_img)

            else:
                return runner.test_imgs(test_dir)

            Log.info("Testing end...")
        except Exception as e:
            Log.error(str(e))
            raise e

    @staticmethod
    def finetune(runner):
        Log.info("Start Finetuning...")
        if runner.configer.get("solver", "lr")["metric"] == "epoch":
            while runner.runner_state["epoch"] < runner.configer.get(
                "solver", "max_epoch"
            ):
                runner.finetune()
        else:
            while runner.runner_state["iters"] < runner.configer.get(
                "solver", "max_iters"
            ):
                runner.finetune()

        Log.info("Finetuning end...")
