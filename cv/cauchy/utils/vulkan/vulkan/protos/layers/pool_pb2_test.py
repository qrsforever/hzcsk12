from __future__ import absolute_import, division, print_function

import pytest

from vulkan.protos.layers import pooling_pb2
from google.protobuf import text_format
import numpy


def test_pool():
  test_pool_def = r"""
  name : "pool1"
  layer_mode :  AVGPOOL2D
  inputs : "layer1"
  outputs : "layer3"
  outputs : "layer4"
  layer_params {
    kernel_size : "2"
    kernel_size : "3"
    stride : "2"
    padding : "0"
    ceil_mode : "False"
    count_include_pad : "Falase"
  }
  layer_builder : "NNTorchLayer"
  """
  proto = pooling_pb2.PoolLayer()
  text_format.Merge(test_pool_def, proto)
  assert proto.name == 'pool1'
  assert proto.layer_mode == pooling_pb2.AVGPOOL2D
  assert proto.inputs == ["layer1"]
  assert proto.outputs == ["layer3", "layer4"]
  assert proto.layer_params.kernel_size == ["2", "3"]
  assert proto.layer_builder == "NNTorchLayer"