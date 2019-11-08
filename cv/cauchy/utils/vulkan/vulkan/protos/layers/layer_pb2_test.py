from __future__ import absolute_import, division, print_function

import pytest

from . import layers_pb2, conv_pb2
from google.protobuf import text_format


def test_layer():
  test_layer_conf = r"""
  conv {
    name : "conv1"
    layer_mode : CONV2D 
    inputs : "layer1"
    inputs : "layer2"
    outputs : "layer3"
    outputs : "layer4"
    layer_params {
      in_channels : "3"
      out_channels : "6"
      kernel_size : "2"
      kernel_size : "3"
    }
    layer_builder : "NNTorchLayer"
  } 
  """
  layer_proto = layers_pb2.Layer()
  text_format.Merge(test_layer_conf, layer_proto)

  # check layer type
  assert layer_proto.WhichOneof("layer_oneof") == 'conv'
  assert layer_proto.conv.name == "conv1"
  assert layer_proto.conv.layer_mode == conv_pb2.CONV2D
  assert layer_proto.conv.layer_builder == "NNTorchLayer"
