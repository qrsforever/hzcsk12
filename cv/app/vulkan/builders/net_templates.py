from __future__ import absolute_import, division, print_function

__all__ = ['PlainNetTemplate', 'ResnetTemplate', 'DensenetTemplate']

PlainNetTemplate = r"""from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import vulkan.layers.core as vlc
import os

class {0}(nn.Module):
  def __init__(self,):
    super({0}, self).__init__()
    {1}
  def forward(self, x):
    {2}

def custom_model():
  return {0}()
"""


class ResnetTemplate():

  @staticmethod
  def basic_block():
    # basic block in resnet
    return r"""class {0}(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super({0}, self).__init__()
    {1}
    self.relu = nn.ReLU(inplace=True)
    self.stride = stride

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_planes,
              self.expansion * planes,
              kernel_size=1,
              stride=stride,
              bias=False), nn.BatchNorm2d(self.expansion * planes))

  def forward(self, x):
    identity = self.shortcut(x)
    {2}

    out = self.relu(out)

    return out
"""

  @staticmethod
  def bottleneck():
    # bottleneck in resnet
    return r"""class {0}(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super({0}, self).__init__()
    {1}
    self.relu = nn.ReLU(inplace=True)
    self.stride = stride

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_planes,
              self.expansion * planes,
              kernel_size=1,
              stride=stride,
              bias=False), nn.BatchNorm2d(self.expansion * planes))


  def forward(self, x):
    identity = self.shortcut(x)
    {2}

    out = self.relu(out)
    return out
"""

  @staticmethod
  def net_arch():
    # positions to be substituted
    # 0 net name
    # 1 number of class to be classified
    # 2 layers before resnet clock
    # 3 layers after resnet clock
    # 4 forward clause for layers before resnet block
    # 5 forward clause for layers after resnet block
    return r"""class {0}(nn.Module):

  def __init__(self, block, num_blocks, num_classes={1}, zero_init_residual=False):
    super({0}, self).__init__()
    self.in_planes = 64
    {2}    
    {3}
    {4}

  def _make_layer(self, block, planes, num_blocks, stride=1):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    # tile up layers
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    {5}
    {6}
    {7}
    return out 
    """

  @staticmethod
  def resnet_bundle():
    # return file templates for whole resnet
    # 0 - block definition
    # 1 - renset definition
    # 3 - function to return a net
    return r"""import torch.nn as nn
import vulkan.layers.core as vlc

{0}

{1}

def custom_model():
  return {2}({3},num_blocks={4}, num_classes={5})"""


class DensenetTemplate():

  @staticmethod
  def dense_bottleneck():
    return r"""class {0}(nn.Module):

  def __init__(self, in_planes, growth_rate):
    super({0}, self).__init__()
    {1}

  def forward(self, x):
    {2}
    out = torch.cat([{3}_feat,x],1)
    return out"""

  @staticmethod
  def densenet_bundle():
    # 0 densennet block
    # 1 netname
    # 2 attr clauses before block
    # 3 attr clauses after block
    # 4 forward clauses before block
    # 5 forward clauses after block
    # 6 return model clauses

    return r"""import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import vulkan.layers.core as vlc
import os

{0}

class Transition(nn.Module):

  def __init__(self, in_planes, out_planes):
    super(Transition, self).__init__()
    self.bn = nn.BatchNorm2d(in_planes)
    self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

  def forward(self, x):
    out = self.conv(F.relu(self.bn(x)))
    out = F.avg_pool2d(out, 2, ceil_mode=True)
    return out


class {1}(nn.Module):

  def __init__(self,
               block,
               nblocks,
               growth_rate=12,
               reduction=0.5,
               num_classes=10):
    super({1}, self).__init__()
    self.growth_rate = growth_rate

    num_planes = 2 * growth_rate
    {2}

    self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
    num_planes += nblocks[0] * growth_rate
    out_planes = int(math.floor(num_planes * reduction))
    self.trans1 = Transition(num_planes, out_planes)
    num_planes = out_planes

    self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
    num_planes += nblocks[1] * growth_rate
    out_planes = int(math.floor(num_planes * reduction))
    self.trans2 = Transition(num_planes, out_planes)
    num_planes = out_planes

    self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
    num_planes += nblocks[2] * growth_rate
    out_planes = int(math.floor(num_planes * reduction))
    self.trans3 = Transition(num_planes, out_planes)
    num_planes = out_planes

    self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
    num_planes += nblocks[3] * growth_rate

    {3}

  def _make_dense_layers(self, block, in_planes, nblock):
    layers = []
    for i in range(nblock):
      layers.append(block(in_planes, self.growth_rate))
      in_planes += self.growth_rate
    return nn.Sequential(*layers)

  def forward(self, x):
    {4}
    out = self.trans1(self.dense1(out))
    out = self.trans2(self.dense2(out))
    out = self.trans3(self.dense3(out))
    out = self.dense4(out)
    {5}
    return out

{6}"""
