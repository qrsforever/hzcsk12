#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Select Det Model for object detection.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from cauchy.models.det.nets.darknet_yolov2 import DarkNetYolov2
from cauchy.models.det.nets.darknet_yolov3 import DarkNetYolov3
from cauchy.models.det.nets.vgg300_ssd import Vgg300SSD
from cauchy.models.det.nets.vgg512_ssd import Vgg512SSD
from cauchy.models.det.nets.faster_rcnn import FasterRCNN
from cauchy.models.det.loss.det_modules import (
    SSDMultiBoxLoss,
    SSDFocalLoss,
    YOLOv3Loss,
    FRLoss,
)
from cauchy.utils.tools.logger import Logger as Log
from cauchy.models.det.nets.custom_backbone_ssd import CustomBackboneSSD

DET_MODEL_DICT = {
    "vgg300_ssd": Vgg300SSD,
    "vgg512_ssd": Vgg512SSD,
    "darknet_yolov2": DarkNetYolov2,
    "darknet_yolov3": DarkNetYolov3,
    "faster_rcnn": FasterRCNN,
}

DET_LOSS_DICT = {
    "ssd_multibox_loss": SSDMultiBoxLoss,
    "ssd_focal_loss": SSDFocalLoss,
    "yolov3_det_loss": YOLOv3Loss,
    "fr_loss": FRLoss,
}


class ModelManager(object):
    def __init__(self, configer):
        self.configer = configer

    def object_detector(self):
        model_name = self.configer.get("network", "model_name")

        if model_name not in DET_MODEL_DICT:
            # QRS: TODO only test
            if model_name.split("_")[0] == "custom":
                model_name = "_".join(model_name.split("_")[1:])
                return CustomBackboneSSD(model_name, self.configer)
            else:
                raise RuntimeError('Model: {} not in dict'.format(model_name))

        model = DET_MODEL_DICT[model_name](self.configer)

        return model

    def get_det_loss(self, loss_type=None):
        key = (
            self.configer.get("loss", "loss_type")
            if loss_type is None
            else loss_type
        )
        if key not in DET_LOSS_DICT:
            Log.error("Loss: {} not valid!".format(key))
            exit(1)

        loss = DET_LOSS_DICT[key](self.configer)
        if (
            self.configer.get("network", "loss_balance")
            and len(range(torch.cuda.device_count())) > 1
        ):
            from extensions.tools.parallel.data_parallel import (
                DataParallelCriterion,
            )

            loss = DataParallelCriterion(loss)

        return loss
