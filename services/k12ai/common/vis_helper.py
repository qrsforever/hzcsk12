#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file vis_helper.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-04-26 16:13


import io
import numpy as np
import torch
import torchvision
import matplotlib.cm as cm
import torch.nn as nn
import matplotlib.pyplot as plt # noqa

from torchvision.utils import make_grid
from torch.nn import functional as F
from .util_misc import handle_exception

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


class VanillaBackpropagation(VisBase):
    def forward(self, images):
        if isinstance(images, (tuple, list)):
            images = torch.stack(images).to(self.device)
        self.images = images.requires_grad_()
        return super(VanillaBackpropagation, self).forward(self.images)

    def generate(self):
        gradient = self.images.grad.clone()
        self.images.grad.zero_()
        return gradient

    @staticmethod
    def mkimage(gradient):
        gradient = gradient.cpu().numpy().transpose(1, 2, 0)
        gradient -= gradient.min()
        gradient /= gradient.max()
        gradient *= 255.0
        return np.uint8(gradient)


class GuidedBackpropagation(VanillaBackpropagation):
    def __init__(self, model):
        super(GuidedBackpropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for name, module in self.model.named_modules():
            self.handlers.append(module.register_backward_hook(backward_hook))


class Deconvnet(VanillaBackpropagation):
    def __init__(self, model):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_out[0]),)

        for name, module in self.model.named_modules():
            self.handlers.append(module.register_backward_hook(backward_hook))


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
        if target_layer in self.fmap_pool.keys():
            fmaps = self.fmap_pool[target_layer]
            grads = self.grad_pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(gcam, size=self.image_shape, mode="bilinear", align_corners=False)

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

    @staticmethod
    def mkimage(gcam_image, raw_image):
        gcam = gcam_image.cpu().numpy()
        cmap = cm.jet_r(gcam)[..., :3] * 255.0
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        return gcam.astype(np.uint8)


class FeatureMaps(VisBase):
    def __init__(self, model, target_layers=None):
        super(FeatureMaps, self).__init__(model)
        self.fmap_pool = {}
        self.target_layers = target_layers

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        for name, module in self.model.named_modules():
            if self.target_layers is None or name in self.target_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))

    def generate(self, nrow=10, resize=(64, 64), rank='TB'):
        features = []
        for name, activation in self.fmap_pool.items():
            if activation.size(1) < nrow:
                nrow = activation.size(1)
            features.append(activation.squeeze().cpu())

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor()
        ])
        
        image_list = []
        if rank == 'LR':
            for idx in range(nrow):
                for img in features:
                    image_list.append(transforms(img[idx]))
            nrow = len(features)
        else:
            for img in features:
                for idx in range(nrow):
                    image_list.append(transforms(img[idx]))
        return make_grid(image_list, nrow=nrow, padding=4, normalize=True)

    @staticmethod
    def mkimage(tensor, figsize=(12, 12)):
        plt.figure(figsize=figsize)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(tensor.permute(1, 2, 0))
        with io.BytesIO() as fw:
            plt.savefig(fw)
            return fw.getvalue()
        return None
        

@handle_exception(handler=print)
def generate_model_graph(module, inputs, fmt='svg'):
    from onnx import ModelProto
    from onnx.tools.net_drawer import GetPydotGraph
    onnx_fp = io.BytesIO()
    torch.onnx.export(
        module,
        inputs,
        onnx_fp,
        export_params=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        verbose=False)
    onnx_fp.seek(0)
    model_proto = ModelProto()
    model_proto.ParseFromString(onnx_fp.read())
    pydot_graph = GetPydotGraph(
        model_proto.graph,
        name=model_proto.graph.name,
        rankdir='TB' # 'LR'
    )
    # output: bytes
    if fmt == 'svg':
        svg = pydot_graph.create_svg().decode()
        return ''.join(svg.split('\n')[6:])
    return io.BytesIO(pydot_graph.create_png()).getvalue()


@handle_exception(handler=print)
def generate_model_autograd(module, inputs, fmt='svg'):
    from torchviz import make_dot
    dot = make_dot(module(inputs), params=dict(module.named_parameters()))
    # output: bytes
    if fmt == 'svg':
        svg = dot.pipe(format='svg').decode()
        return ''.join(svg.split('\n')[6:])
    return dot.pipe(format='png')
