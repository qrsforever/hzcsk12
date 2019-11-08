#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Class Definition for Image Classifier.

from __future__ import absolute_import, division, print_function

import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

from cauchy.datasets.cls.data_loader import DataLoader
from cauchy.methods.tools.blob_helper import BlobHelper
from cauchy.methods.tools.runner_helper import RunnerHelper
from cauchy.models.cls.model_manager import ModelManager
from cauchy.utils.helpers.image_helper import ImageHelper
from cauchy.utils.helpers.json_helper import JsonHelper
from cauchy.utils.parser.cls_parser import ClsParser
from cauchy.utils.tools.logger import Logger as Log
from cauchy.metrics.cls import ClsRunningScore
from cauchy.utils.tools.res_filter import partial_list_items, select_samples


class FCClassifierTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.cls_model_manager = ModelManager(configer)
        self.cls_data_loader = DataLoader(configer)
        self.cls_parser = ClsParser(configer)
        self.cls_running_score = ClsRunningScore(configer)
        self.device = torch.device(
            "cpu" if self.configer.get("gpu") is None else "cuda"
        )
        self.cls_net = None
        if self.configer.get("dataset") == "imagenet":
            with open(
                os.path.join(
                    self.configer.get("project_dir"),
                    "datasets/cls/imagenet/imagenet_class_index.json",
                )
            ) as json_stream:
                name_dict = json.load(json_stream)
                name_seq = [
                    name_dict[str(i)][1]
                    for i in range(self.configer.get("data", "num_classes"))
                ]
                self.configer.add(["details", "name_seq"], name_seq)
        self.runner_state = dict()

        self._init_model()

    def _init_model(self):
        self.cls_net = self.cls_model_manager.image_classifier()
        self.cls_net = RunnerHelper.load_net(self, self.cls_net)
        self.cls_net.eval()

    def test_img(self, image_path):
        # Log.info('Image Path: {}'.format(image_path))
        img = ImageHelper.read_image(
            image_path,
            tool=self.configer.get("data", "image_tool"),
            mode=self.configer.get("data", "input_mode"),
        )

        trans = None
        if self.configer.get("dataset") == "imagenet":
            if self.configer.get("data", "image_tool") == "cv2":
                img = Image.fromarray(img)

            trans = transforms.Compose(
                [transforms.Scale(256), transforms.CenterCrop(224)]
            )
        else:
            trans = transforms.Compose([])

        assert trans is not None
        img = trans(img)

        inputs = self.blob_helper.make_input(
            img, input_size=self.configer.get("test", "input_size"), scale=1.0
        )

        with torch.no_grad():
            outputs = self.cls_net(inputs)

        json_dict = {}
        outputs = nn.functional.softmax(outputs, dim=1).cpu().data.numpy()
        json_dict["probs"] = outputs.tolist()[0]

        Log.info("Testing end...")
        return json.dumps(json_dict)

    def test_imgs(self, test_dir):
        test_loader = self.cls_data_loader.get_testloader(test_dir)
        labels_list = []
        outputs_list = []
        # relative path to each file
        rltv_path_list = []
        res_dict = {}
        try:
            with torch.no_grad():
                for data_dict in test_loader:
                    # load data
                    inputs = data_dict["img"]
                    labels = data_dict["label"]
                    img_paths = data_dict["img_path"]

                    inputs, labels = RunnerHelper.to_device(
                        self, inputs, labels
                    )

                    outputs = self.cls_net(inputs)
                    outputs = RunnerHelper.gather(self, outputs)
                    outputs = nn.functional.softmax(outputs, dim=1)
                    self.cls_running_score.update(outputs, labels)

                    labels_list.append(labels.cpu().numpy())
                    outputs_list.append(outputs.cpu().numpy())
                    rltv_path_list.extend(img_paths)
            labels = np.hstack(labels_list)
            outputs = np.vstack(outputs_list)
            top1_pred, top3_pred, top5_pred = self.cls_running_score.get_pred()
            labels = labels.tolist()
            outputs = outputs.tolist()
            selected_indices = select_samples(labels, 10)
            selected_labels = partial_list_items(labels, selected_indices)
            selected_outputs = partial_list_items(outputs, selected_indices)
            selected_rltv_path_list = partial_list_items(
                rltv_path_list, selected_indices
            )
            res_dict = {
                "labels": selected_labels,
                "outputs": selected_outputs,
                "rltv_path_list": selected_rltv_path_list,
                "acc_top_1": float(accuracy_score(labels, top1_pred)),
                "acc_top_3": float(accuracy_score(labels, top3_pred)),
                "acc_top_5": float(accuracy_score(labels, top5_pred)),
                "confusion_mat_1": confusion_matrix(labels, top1_pred).tolist(),
                "confusion_mat_3": confusion_matrix(labels, top3_pred).tolist(),
                "confusion_mat_5": confusion_matrix(labels, top5_pred).tolist(),
            }
            Log.info("Testing end...")
        except Exception as e:
            Log.info(str(e))
            raise e
        return json.dumps(
            res_dict, separators=(",", ":"), sort_keys=True, indent=4
        )

    def debug(self, vis_dir):
        count = 0
        for i, data_dict in enumerate(self.cls_data_loader.get_trainloader()):
            inputs = data_dict["img"]
            labels = data_dict["label"]
            eye_matrix = torch.eye(self.configer.get("data", "num_classes"))
            labels_target = eye_matrix[labels.view(-1)].view(
                inputs.size(0), self.configer.get("data", "num_classes")
            )

            for j in range(inputs.size(0)):
                count = count + 1
                if count > 20:
                    exit(1)

                ori_img_bgr = self.blob_helper.tensor2bgr(inputs[j])

                json_dict = self.__get_info_tree(labels_target[j])
                image_canvas = self.cls_parser.draw_label(
                    ori_img_bgr.copy(), json_dict["label"]
                )

                cv2.imwrite(
                    os.path.join(vis_dir, "{}_{}_vis.png".format(i, j)),
                    image_canvas,
                )
                cv2.imshow("main", image_canvas)
                cv2.waitKey()
