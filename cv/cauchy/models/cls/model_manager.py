#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Select Cls Model for pose detection.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from cauchy.models.cls.loss.cls_modules import FCCELoss, FCCenterLoss
from cauchy.models.cls.nets import (
    alexnet,
    gen_densenet,
    gen_resnet,
    inception_v3,
    gen_vggnet,
    gen_squeezenet,
    gen_shufflenetv2,
)
from cauchy.utils.tools.load_custom_model import load_model
from cauchy.utils.tools.logger import Logger as Log


class ModelManager(object):
    def __init__(self, configer):
        self.configer = configer
        self.CLS_MODEL_DICT = {
            "resnet": gen_resnet,
            "densenet": gen_densenet,
            "alexnet": alexnet,
            "inception": inception_v3,
            "vgg": gen_vggnet,
            "squeezenet": gen_squeezenet,
            "shufflenetv2": gen_shufflenetv2,
        }
        self.CLS_LOSS_DICT = {
            "fc_ce_loss": FCCELoss,
            "fc_center_loss": FCCenterLoss,
        }

    def freeze_layers(self, model, layer_list):
        """free layers which are not to be freezed
    Args:
      model (nn.moudel): a predefined model
      layer_list (list): list of layers to requires update
    """
        for layer_name, layer_param in model.named_parameters():
            if layer_name not in layer_list:
                layer_param.required_grad = False

    def image_classifier(self,):
        model_name = self.configer.get("network", "model_name")
        pretrained = self.configer.get("network", "pretrained")
        num_classes = self.configer.get("data", "num_classes")
        proj_dir = self.configer.get("project_dir")

        if model_name.split("_")[0] == "custom":
            model_name = "_".join(model_name.split("_")[1:])
            return load_model(proj_dir, model_name)
        elif model_name.split("_")[0] not in self.CLS_MODEL_DICT.keys():
            Log.error("Model: {} not valid!".format(model_name))
            exit(1)

        if pretrained:
            if self.configer.get("phase") == "train":
                # get model to be fine tuned
                weights_host = self.configer.get("network", "weights_host")
                unfreezed_layers = self.configer.get(
                    "network", "unfreezed_layers"
                )
                return self._load_pretrained_model(
                    model_name, weights_host, unfreezed_layers, num_classes
                )
        else:
            model_type = model_name.split("_")[0]
            kwargs = {
                "model_name": model_name,
                "num_classes": num_classes,
                "pretrained": False,
            }
            if model_type in ["mobilenetv2", "shufflenetv2"]:
                try:
                    width_mult = self.configer.get("network", "width_mult")
                except Exception as e:
                    Log.info(
                        "params width_mult should be provided for {}".format(
                            model_type
                        )
                    )
                    raise e
                kwargs["width_mult"] = float(width_mult)

            model = self.CLS_MODEL_DICT[model_type](**kwargs)
            if self.configer.get("phase") == "test":
                self._adapt_models(model, model_name, num_classes)
            return model

    def _adapt_resnet(self, model, num_classes):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        Log.info(
            "resent pretrained model adapted for {} classes".format(num_classes)
        )

    def _adapt_shufflenet(self, model, num_classes):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        Log.info(
            "shufflnet pretrained model adapted for {} classes".format(
                num_classes
            )
        )

    def _adapt_alexnet(self, model, num_classes):
        model.classifier[6] = nn.Linear(4096, num_classes)
        Log.info(
            "alexnet pretrained model adapted for {} classes".format(
                num_classes
            )
        )

    def _adapt_densenet(self, model, num_classes):
        in_features = {
            "densenet_121": 1024,
            "densenet_161": 2208,
            "densenet_169": 1664,
            "densenet_201": 1920,
        }
        if self.configer.get("network", "model_name") in in_features:
            model.classifier = nn.Linear(
                in_features[self.configer.get("network", "model_name")],
                num_classes,
            )
            Log.info(
                "densenet pretrained model adapted for {} classes".format(
                    num_classes
                )
            )
        else:
            Log.error("Invalid densetnet architecture")
            exit(1)

    def _adapt_inception(self, model, num_classes):
        model.AuxLogits.fc = nn.Linear(768, num_classes)
        model.fc = nn.Linear(2048, num_classes)
        Log.info(
            "inception pretrained model adapted for {} classes".format(
                num_classes
            )
        )

    def _adapt_vgg(self, model, num_classes):
        model.classifier[6] = nn.Linear(4096, num_classes)
        Log.info(
            "vgg model pretrained adapted for {} classes".format(num_classes)
        )

    def _adapt_squuezenet(self, model, num_classes):
        model.num_classes = num_classes
        model.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        Log.info(
            "squeeze model pretrained adapted for {} classes".format(
                num_classes
            )
        )

    def _adapt_models(self, model, model_name, num_classes):
        """adapt models according to num_classes
    
    Args:
      model (nn.module): a model
      model_name (str): str represents the family of model
      num_classes (int): number of output classes
    
    Raises:
      NotImplementedError: Error
    """

        if model_name.split("_")[0] == "resnet":
            self._adapt_resnet(model, num_classes)
        elif model_name.split("_")[0] == "densenet":
            self._adapt_densenet(model, num_classes)
        elif model_name == "alexnet":
            self._adapt_alexnet(model, num_classes)
        elif model_name == "inception":
            self._adapt_inception(model, num_classes)
        elif model_name.split("_")[0] == "vgg":
            self._adapt_vgg(model, num_classes)
        elif model_name.split("_")[0] == "squeezenet":
            self._adapt_squuezenet(model, num_classes)
        elif model_name.split("_")[0] == "shufflenetv2":
            self._adapt_shufflenet(model, num_classes)
        else:
            raise NotImplementedError

    def _load_pretrained_model(
        self, model_name, weights_host, unfreezed_layers, num_classes
    ):
        """load pretrained model and freeze layers except unfreezed layers
    
    Args:
      model_name (str): name of pretrained model
      unfreezed_layers (list): list of layers not to be freezed
      num_classes (int): number of classes  to classify
    
    Returns:
      [nn.mudule]: pretrained model with layers partially freezed
    """
        # get pretrained model
        model_type = model_name.split("_")[0]
        # load weights
        model = self.CLS_MODEL_DICT[model_type](
            model_name=model_name, pretrained=True, weights_host=weights_host
        )
        # freeze partial layers
        if unfreezed_layers is not None:
            self.freeze_layers(model, unfreezed_layers)
            self._adapt_models(model, model_name, num_classes)

        return model

    def get_cls_loss(self, loss_type=None):
        key = (
            self.configer.get("loss", "loss_type")
            if loss_type is None
            else loss_type
        )
        if key not in self.CLS_LOSS_DICT:
            Log.error("Loss: {} not valid!".format(key))
            exit(1)

        loss = self.CLS_LOSS_DICT[key](self.configer)
        if (
            self.configer.get("network", "loss_balance")
            and len(range(torch.cuda.device_count())) > 1
        ):
            device_ids = self.configer.get("gpu")
            from cauchy.extensions.tools.parallel.data_parallel import (
                DataParallelCriterion,
            )

            loss = DataParallelCriterion(loss, device_ids=device_ids)

        return loss
