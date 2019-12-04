from __future__ import absolute_import, division, print_function

import pytest

from vulkan.protos.skeletons import plain_pb2
from google.protobuf import text_format


def test_plainnet():
  """ test plain net proto
  test content:
  1. net name
  2. whether multiple layer can be parse successfully
  """
  test_plainnet_conf = r"""
  name : "alexnet"
  layer : {
    conv: {
      name : "conv1"
    } 
  }
  layer : {
    pool: {
      name : "pool1"
    } 
  }
  """
  plainnet_proto = plain_pb2.PlainNet()
  text_format.Merge(test_plainnet_conf, plainnet_proto)
  assert plainnet_proto.name == "alexnet"
  assert len(plainnet_proto.layer) == 2
  assert plainnet_proto.layer[0].WhichOneof("layer_oneof") == 'conv'
  assert plainnet_proto.layer[1].WhichOneof("layer_oneof") == 'pool'
  assert plainnet_proto.layer[0].conv.name == "conv1"