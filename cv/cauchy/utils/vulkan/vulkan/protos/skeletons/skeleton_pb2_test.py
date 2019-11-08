from __future__ import absolute_import, division, print_function

import pytest

from vulkan.protos.skeletons import skeleton_pb2
from google.protobuf import text_format


def test_plain_skeleton():
  test_plain_skeleton_conf = r"""
  plain_net {
    name : "alexnet"
    layer  {
      conv {
        name : "conv1"
      } 
    }
    layer  {
      pool {
        name : "pool1"
      } 
    } 
  }
  """
  skeleton_proto = skeleton_pb2.skeletons()
  text_format.Merge(test_plain_skeleton_conf, skeleton_proto)

  assert skeleton_proto.WhichOneof("skeleton_oneof") == "plain_net"
  assert skeleton_proto.plain_net.name == "alexnet"
