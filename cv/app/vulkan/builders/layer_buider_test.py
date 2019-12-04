from __future__ import absolute_import, division, print_function

import pytest
from google.protobuf import text_format

from vulkan.builders.layer_builder import LayerBuilder
from vulkan.protos.layers import layers_pb2


@pytest.fixture
def layer_generator():
  """return a layer factory instance"""
  return LayerBuilder()


test_protos = (
    # conv
    r"""
  conv {
    name : "conv1"
    layer_mode : CONV2D 
    inputs : "layer1"
    outputs : "layer3"
    layer_params {
      in_channels : "3"
      out_channels : "6"
      kernel_size : "2"
      kernel_size : "3"
      bias : "True"

    }
    layer_builder : "NNTorchLayer"
  }
  
  """,
    # pool
    r"""
  pool {
    name : "pool1"
    layer_mode : MAXPOOL2D
    inputs : "layer2"
    outputs : "layer3"
    layer_params {
      kernel_size : "2"
      kernel_size : "2"
      padding : "1"
      return_indices : "False"
    }
    layer_builder : "NNTorchLayer"
  }
  """,
    # linear layer
    r"""
  linear {
    name : "linear1"
    inputs : "layer1"
    outputs : "layer2"
    layer_mode : LINEAR
    layer_params {
      in_features : "20"
      out_features : "30"
      bias : "False"
    }
    layer_builder : "NNTorchLayer"
  }
  """,
    # dropout layer
    r"""
  dropout {
    name : "dropout1"
    inputs : "layer1"
    outputs : "layer2"
    layer_mode : DROPOUT
    layer_params {
      inplace : "True"
      p : "0.6"
    }
    layer_builder : "NNTorchLayer"
  }
  """,
    # RELU
    r"""
  act {
    name : "relu1"
    inputs : "layer1"
    outputs : "layer2"
    layer_mode : RELU
    layer_params {
      inplace : "False"
    }
    layer_builder : "NNTorchLayer"
  }
  """,
    # reshape
    r"""
  vulkan {
    name : "reshape1"
    inputs : "layer1"
    layer_mode : RESHAPE
    layer_params {
      target_shape : "4096"
    }
    layer_builder : "NNTorchLayer"
  }
  """,
    # cat
    r"""
  basefunc {
    name : "cat1"
    inputs : "layer1"
    inputs : "layer2"
    layer_mode : CAT
    layer_params {
      dim : "0"
    }
    layer_builder : "TorchFuncLayer"
  }
  """,
    # norm layer
    r"""
  norm {
    name : "bn1"
    inputs : "layer1"
    outputs : "layer2"
    layer_mode : BATCHNORM2D
    layer_params {
      num_features : "36"
    }
    layer_builder : "NNTorchLayer"
  }
  """,
    # flatten
    r"""
  vulkan {
    name : "flatten1"
    inputs : "layer1"
    outputs : "layer2"
    layer_mode : FLATTEN
    layer_params : {
      start_dim : "1"
      end_dim : "-1"
    }
    layer_builder : "NNTorchLayer"
  }
  """,
    # add
    r"""
  basefunc {
    name : "add1"
    inputs : "layer1"
    inputs : "layer2"
    layer_mode : ADD
    layer_builder : "TorchFuncLayer"
  }
  """,
    # zero padding
    r"""
  padding {
    name : "pad1"
    inputs : "layer1"
    layer_mode : ZEROPAD2D
    layer_params : {
      padding : "1"
      padding : "1"
    }
    layer_builder : "NNTorchLayer"
  }
  """,
)

layer_types = ('conv', 'pool', 'linear', 'dropout', 'relu', 'reshape', 'cat',
               'norm', 'flatten', 'add', 'padding')

expected_attr_clause = (
    # conv
    r"""self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(2,3), bias=True)""",
    # pool
    r"""self.pool1 = nn.MaxPool2d(kernel_size=(2,2), padding=1, return_indices=False)""",
    # linear
    r"""self.linear1 = nn.Linear(in_features=20, out_features=30, bias=False)""",
    # dropout
    r"""self.dropout1 = nn.Dropout(inplace=True, p=0.6)""",
    # relu
    r"""self.relu1 = nn.ReLU(inplace=False)""",
    # reshape
    r"""self.reshape1 = vlc.Reshape(target_shape=(4096,))""",
    # cat
    r"""""",
    # norm
    r"""self.bn1 = nn.BatchNorm2d(num_features=36)""",
    # flatten
    r"""self.flatten1 = vlc.Flatten(start_dim=1, end_dim=-1)""",
    # add
    r"""""",
    # padding
    r"""self.pad1 = nn.ZeroPad2d(padding=(1,1))""",
)

expected_forward_clause = (
    # conv
    r"""conv1_feat = self.conv1(layer1_feat)""",
    # pool
    r"""pool1_feat = self.pool1(layer2_feat)""",
    # linear
    r"""linear1_feat = self.linear1(layer1_feat)""",
    # dropout
    r"""dropout1_feat = self.dropout1(layer1_feat)""",
    # relu
    r"""relu1_feat = self.relu1(layer1_feat)""",
    # reshape
    r"""reshape1_feat = self.reshape1(layer1_feat)""",
    # cat
    r"""cat1_feat = torch.cat((layer1_feat, layer2_feat), dim=0)""",
    # norm
    r"""bn1_feat = self.bn1(layer1_feat)""",
    # flatten
    r"""flatten1_feat = self.flatten1(layer1_feat)""",
    # add
    r"""add1_feat = torch.add(layer1_feat, layer2_feat)""",
    # padding
    r"""pad1_feat = self.pad1(layer1_feat)""",
)

test_instances = [
    (proto, layer, attr_clause, forward_clause)
    for proto, layer, attr_clause, forward_clause in zip(
        test_protos, layer_types, expected_attr_clause, expected_forward_clause)
]

test_pipe = "proto_text, layer, attr_clause, forward_clause"


def extract_params(clause):
  first_parentheses = clause.find('(')
  pre_parentheses = clause[:first_parentheses]
  params = clause[first_parentheses + 1:-1].split(', ')
  return pre_parentheses, params


def check_params(params_1, params_2):
  if type(params_1) != type(params_2):
    return False
  elif len(params_1) != len(params_2):
    return False
  else:
    for k_v in params_1:
      if k_v not in params_2:
        return False
  return True


def check_clause(gen_clause, expected_clause):
  if expected_clause == "" and gen_clause == "":
    return True
  else:
    gen_pre_paren, gen_params = extract_params(gen_clause)
    expected_pre_paren, expected_params = extract_params(expected_clause)
    if gen_pre_paren != expected_pre_paren:
      return False

    else:
      return check_params(gen_params, expected_params)


@pytest.mark.parametrize(test_pipe, test_instances)
def test_layers(layer_generator, proto_text, layer, attr_clause,
                forward_clause):
  # read layer proto
  layer_proto = layers_pb2.Layer()
  print(layer_proto)
  text_format.Merge(proto_text, layer_proto)
  # create layer
  layer_obj = layer_generator.create_layer(layer_proto)
  # test python clause
  # assert layer_obj.attributes_clause() == attr_clause
  # assert layer_obj.forward_clause() == forward_clause
  assert check_clause(layer_obj.attributes_clause(), attr_clause)
  assert check_clause(layer_obj.forward_clause(), forward_clause)
