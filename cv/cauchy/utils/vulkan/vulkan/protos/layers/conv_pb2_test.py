from __future__ import absolute_import, division, print_function

import pytest

from vulkan.protos.layers import conv_pb2
from google.protobuf import text_format
import numpy


def test_conv():
  test_conv_def = r"""
  name : "conv1"
  layer_mode : CONV2D 
  inputs : "layer1"
  outputs : "layer3"
  outputs : "layer4"
  layer_params {
    in_channels : "3"
    out_channels : "6"
    kernel_size : "2"
    kernel_size : "3"

  }
  layer_builder : "NNTorchLayer"
  """
  proto = conv_pb2.ConvLayer()
  text_format.Merge(test_conv_def, proto)
  assert proto.name == 'conv1'
  assert proto.layer_mode == conv_pb2.CONV2D
  assert proto.layer_params.in_channels == "3"
  assert proto.layer_params.out_channels == "6"
  assert proto.layer_params.kernel_size == ["2", "3"]
  assert proto.inputs == ["layer1"]
  assert proto.outputs == ["layer3", "layer4"]
