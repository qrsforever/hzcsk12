from __future__ import absolute_import, division, print_function

import importlib.util
import tempfile

import pytest
import torch
from google.protobuf import text_format

from vulkan.protos.skeletons import skeleton_pb2
from vulkan.utils.pb import read_proto

from .net_builder import NetBuilder


def test_plainnet_builder():
  """ test net buidler"""

  net_def = r"""
plain_net {
name:"Alexnet"

layer{
  conv{
    name : "Conv2d_256"
    layer_builder : "NNTorchLayer"
    layer_mode : CONV2D
    layer_params:{
      in_channels : "3"
      out_channels : "64"
      kernel_size : "11"
      stride : "4"
      padding : "2"
    }
    inputs: "x"
        outputs:"Relu_510"
  }
}
layer{
  act{
    name : "Relu_510"
    layer_builder : "NNTorchLayer"
    layer_mode : RELU
    layer_params:{
      inplace : "True"
    }
    inputs:"Conv2d_256"
    outputs:"Conv2d_773"
outputs:"Conv2d_229"
  }
}
layer{
  conv{
    name : "Conv2d_773"
    layer_builder : "NNTorchLayer"
    layer_mode : CONV2D
    layer_params:{
      in_channels : "64"
      out_channels : "128"
      kernel_size : "5"
      padding : "2"
    }
    inputs:"Relu_510"
    outputs:"Reshape_264"
  }
}
layer{
  conv{
    name : "Conv2d_229"
    layer_builder : "NNTorchLayer"
    layer_mode : CONV2D
    layer_params:{
      in_channels : "64"
      out_channels : "128" 
      kernel_size : "5"
      padding : "2"
    }
    inputs:"Relu_510"
    outputs:"Reshape_264"
  }
}
layer{
  basefunc{
    name : "Reshape_264"
    layer_builder : "TorchFuncLayer"
    layer_mode : CAT
    layer_params:{
      dim:"0"
    }
    inputs:"Conv2d_773"
    inputs:"Conv2d_229"
  }
}
}
  """

  with tempfile.TemporaryDirectory() as tmp_dir:
    # net_def = text_format.Merge(net_def, skeleton_pb2.skeletons())
    # net_def = eval("net_def.{}".format(net_def.WhichOneof("skeleton_oneof")))
    # net_def = read_proto(net_def, skeleton_pb2.skeletons(), "skeleton_oneof")
    # net_buider = PlainNetBuilder(net_def)
    net = NetBuilder(net_def)
    net = net.build_net()
    net.write_net("{}/test_net.py".format(tmp_dir))
    with open("{}/test_net.py".format(tmp_dir)) as fin:
      for each_line in fin:
        print(each_line)
    spec = importlib.util.spec_from_file_location(
        "test_net", "{}/test_net.py".format(tmp_dir))
    test_net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_net)
    test_net_instance = test_net.Alexnet()
    test_input = torch.rand((8, 3, 32, 32))
    test_net_instance(test_input)
