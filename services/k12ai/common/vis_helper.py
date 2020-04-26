#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file vis_helper.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-04-26 16:13


import numpy as np
import torch
import matplotlib.cm as cm

from torch.nn import functional as F

AA = 100


### CNN Visualization

class VisBase(object): # noqa
    def __init__(self, model):
        super(VisBase, self).__init__()
        self.handlers = []
        self.device = next(model.parameters()).device
        self.model = model
        self.model.eval() # TODO

    def forward(self, images):
        if isinstance(images, (tuple, list)):
            images = torch.stack(images).to(self.device)
        self.image_shape = images.shape[2:] # (B, C, W, H)
        self.logits = self.model(images)
        print(self.logits.argmax(dim=1))
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()


class GradCAM(VisBase):
    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(gcam, size=self.image_shape, mode="bilinear", align_corners=False)

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam # regions

    @staticmethod
    def mkimage(gcam_image, raw_image):
        gcam = gcam_image.cpu().numpy()
        cmap = cm.jet_r(gcam)[..., :3] * 255.0
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        return gcam.astype(np.uint8) # Image.fromarray(gcam.astype(np.uint8))
