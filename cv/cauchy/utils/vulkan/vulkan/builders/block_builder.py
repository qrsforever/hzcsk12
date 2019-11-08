from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from google.protobuf import text_format

from ..protos.blocks import blocks_pb2
from .layer_builder import LayerBuilder
from .net_templates import *


class Basic_Block_Builder():
  """basic block builder"""

  def __init__(self, block_proto):
    self.proto = block_proto
    self.layers = self._parse_layers()

  def _parse_layers(self,):
    """parse layers in blocks"""
    layer_generator = LayerBuilder()
    layers = []
    for layer_proto in self.proto.layer:
      layers.append(layer_generator.create_layer(layer_proto))
    return layers

  def _get_attr_clauses(self,):
    """generate attribute clauses from all layers"""
    attr_clauses = [layer.attributes_clause() + '\n' for layer in self.layers]
    return "    ".join(attr_clauses)

  def _get_forward_clauses(self,):
    pass

  def gen_block(self,):
    pass


class Res_Block_Builder(Basic_Block_Builder):
  """block builder for resnet"""

  def __init__(self, block_proto):
    super(Res_Block_Builder, self).__init__(block_proto)

  def _get_forward_clauses(self):
    """genereate foward clauses"""
    f_clauses = [layer.forward_clause() for layer in self.layers]
    f_clauses[0] = self.layers[0].first_forward_clause()
    f_clauses = [f_clause + '\n' for f_clause in f_clauses]
    f_clauses.append('out = {}_feat + identity'.format(
        self.layers[-1].proto.name))
    return "    ".join(f_clauses)

  def gen_block(self,):
    """generate block codes based on the block def"""
    if self.proto.block_mode == blocks_pb2.RES_BASIC:
      block_templates = ResnetTemplate.basic_block()
    else:
      block_templates = ResnetTemplate.bottleneck()

    return block_templates.format(self.proto.name, self._get_attr_clauses(),
                                  self._get_forward_clauses())


class Dense_Block_Builder(Basic_Block_Builder):
  """ block builder for densenet"""

  def __init__(self, block_proto):
    super(Dense_Block_Builder, self).__init__(block_proto)

  def _get_forward_clauses(self):
    """generate foward clauses for densenet"""
    f_clauses = [layer.forward_clause() for layer in self.layers]
    f_clauses[0] = self.layers[0].first_forward_clause()
    f_clauses = [f_clause + '\n' for f_clause in f_clauses]
    return "    ".join(f_clauses)

  def gen_block(self):
    block_template = DensenetTemplate.dense_bottleneck()
    return block_template.format(self.proto.name, self._get_attr_clauses(),
                                 self._get_forward_clauses(),
                                 self.layers[-1].proto.name)


class Block_Builder():
  """generate blocks for Pytorch"""

  def __init__(self, block_proto):
    self.block_proto = block_proto
    self.block_builders = {
        blocks_pb2.RES_BASIC: Res_Block_Builder,
        blocks_pb2.RES_BOTTLENECK: Res_Block_Builder,
        blocks_pb2.DENSE_BOTTLENECK: Dense_Block_Builder
    }

  def build_block(self):
    try:
      return self.block_builders[self.block_proto.block_mode](self.block_proto)
    except KeyError:
      print("block type not supported")
      raise
