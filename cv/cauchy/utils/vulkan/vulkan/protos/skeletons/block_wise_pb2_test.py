from __future__ import absolute_import, division, print_function

import importlib.util
import tempfile

import torch
from google.protobuf import text_format

from vulkan.builders.net_builder import NetBuilder
from vulkan.protos.skeletons import skeleton_pb2
from vulkan.protos.skeletons.block_wise_pb2 import BlockWiseNet
from vulkan.utils.pb import read_proto


def test_resnet():
  net_def = r"""
  block_wise_net {
    name : "resnet"
    net_mode : RESNET
    seq_net {
      name : "before_block"
      layer {
        conv {
          name : "conv1"
          layer_mode : CONV2D
          layer_params : {
            in_channels : "3"
            out_channels : "64"
            kernel_size : "7"
            stride : "2"
            padding : "3"
            bias : "True"
          }
          layer_builder : "NNTorchLayer"
        } 
      }

      layer {
        norm {
          name : "bn1"
          layer_mode : BATCHNORM2D
          layer_params : {
            num_features : "64"
          }
          layer_builder : "NNTorchLayer"
        } 
      }

      layer {
        act {
          name : "relu"
          layer_mode : RELU
          layer_params {
            inplace : "True"
          }
          layer_builder : "NNTorchLayer"
        }
      }

      layer { 
        pool {
          name : "maxpool"
          layer_mode : MAXPOOL2D
          layer_params : {
            kernel_size : "3"
            stride : "2"
            padding : "1"
          }
          layer_builder : "NNTorchLayer"
        }
      }
    }

    block {
        name : "Bottleneck"
        block_mode : RES_BOTTLENECK 
        layer {
          conv {
            name : "conv1"
            inputs : "layer1"
            outputs : "layer2"
            layer_mode : CONV2D
            layer_params {
              in_channels : "in_planes"
              out_channels : "planes"
              kernel_size : "3"
              stride : "1"
              padding : "1"
              bias : "True"
            }
            layer_builder : "NNTorchLayer"
          }
        }

        layer {
          norm {
            name : "bn1"
            inputs : "conv1"
            outputs : "layer2"
            layer_mode : BATCHNORM2D
            layer_params {
              num_features : "planes" 
            }
            layer_builder : "NNTorchLayer"
          }
        }

        layer {
          act {
            inputs : "bn1"
            name : "relu1"
            layer_mode : RELU
            layer_params {
              inplace : "True"
            }
            layer_builder : "NNTorchLayer"
          }
        }

        layer {
          conv {
            name : "conv2"
            inputs : "relu1"
            layer_mode : CONV2D
            layer_params {
              in_channels : "planes"
              out_channels : "planes"
              kernel_size : "3"
              stride : "stride"
              padding : "1"
              bias : "True"
            }
            layer_builder : "NNTorchLayer"
          }
        }

        layer {
          norm {
            name : "bn2"
            inputs : "conv2"
            outputs : "layer2"
            layer_mode : BATCHNORM2D
            layer_params {
              num_features : "planes" 
            }
            layer_builder : "NNTorchLayer"
          }
        }

        layer {
          act {
            inputs : "bn2"
            name : "relu2"
            layer_mode : RELU
            layer_params {
              inplace : "True"
            }
            layer_builder : "NNTorchLayer"
          }
        }

        layer {
          conv {
            name : "conv3"
            inputs : "relu2"
            layer_mode : CONV2D
            layer_params {
              in_channels : "planes"
              out_channels : "planes * self.expansion"
              kernel_size : "3"
              stride : "1" 
              padding : "1"
              bias : "True"
            }
            layer_builder : "NNTorchLayer"
          }
        }

        layer {
          norm {
            name : "bn3"
            inputs : "conv3"
            outputs : "layer2"
            layer_mode : BATCHNORM2D
            layer_params {
              num_features : "planes * self.expansion" 
            }
            layer_builder : "NNTorchLayer"
          }
        }

    }

    seq_net {
      name : "after_block"

      layer : {
        pool : {
          name : "avgpool"
          layer_mode : ADAPTIVEAVGPOOL2D
          layer_params : {
            output_size : "1"
            output_size : "1"
          }
          layer_builder : "NNTorchLayer"
        }
      }

      layer {
        vulkan {
          name : "flatten"
          layer_mode : FLATTEN
          layer_builder : "NNTorchLayer"
          layer_params {
            start_dim : "1"
          }
        }
      }

      layer {
        linear {
          name : "fc"
          layer_mode : LINEAR
          layer_params {
            in_features : "512 * block.expansion"
            out_features : "num_classes" 
          }
          layer_builder : "NNTorchLayer"
        } 
      }

    }

    net_params {
      num_blocks : 3
      num_blocks  : 4
      num_blocks  : 6
      num_blocks  : 3

      planes : 64
      planes : 128
      planes : 256
      planes : 512

      strides : 1
      strides : 2
      strides : 2
      strides : 2

      num_classes : 10
    }
  }

  """
  # tmp_dir = '/tmp'
  with tempfile.TemporaryDirectory() as tmp_dir:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net_builder = NetBuilder(net_def)
    net = net_builder.build_net()
    net.write_net("{}/test_net.py".format(tmp_dir))
    with open("{}/test_net.py".format(tmp_dir)) as fin:
      for each_line in fin:
        print(each_line)
    spec = importlib.util.spec_from_file_location(
        "test_net", "{}/test_net.py".format(tmp_dir))
    test_net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_net)
    test_net_instance = test_net.custom_model().to(device)
    test_input = torch.rand((4, 3, 224, 224)).to(device)
    out_tensor = test_net_instance(test_input)
    assert out_tensor.size() == torch.Size([4, 10])


def test_densenet():
  net_def = r"""
  block_wise_net {
    name : "densenet"
    net_mode : DENSENET

    seq_net {
      name : "before_block" 
      layer{
        conv{
          name : "Conv2d_5905"
          layer_builder : "NNTorchLayer"
          layer_mode : CONV2D
          layer_params {
            in_channels : "3"
            out_channels : "num_planes"
            kernel_size : "3"
            stride : "1"
            padding : "1"
            dilation : "1"
            groups : "1"
            bias : "False"
          }
        }
      }
    }

    block {
      name : "denseblock"
      block_mode : DENSE_BOTTLENECK
      layer{
        norm{
          name : "BatchNorm2d_1221"
          layer_builder : "NNTorchLayer"
          layer_mode : BATCHNORM2D
          outputs : "Relu_3523"
          layer_params {
            num_features : "in_planes"
            eps : "1e-05"
            momentum : "0.1"
            affine : "True"
            track_running_stats : "True"
          }
        }
      }
      layer{
        act{
          name : "Relu_3523"
          layer_builder : "NNTorchLayer"
          layer_mode : RELU
          inputs : "BatchNorm2d_1221"
          outputs : "Conv2d_3988"
          layer_params {
            inplace : "False"
          }
        }
      }
      layer{
        conv{
          name : "Conv2d_3988"
          layer_builder : "NNTorchLayer"
          layer_mode : CONV2D
          inputs : "Relu_3523"
          outputs : "BatchNorm2d_8569"
          layer_params {
            in_channels : "in_planes"
            out_channels : "4 * growth_rate"
            kernel_size : "1"
            stride : "1"
            padding : "0"
            dilation : "1"
            groups : "1"
            bias : "False"
          }
        }
      }
      layer{
        norm{
          name : "BatchNorm2d_8569"
          layer_builder : "NNTorchLayer"
          layer_mode : BATCHNORM2D
          inputs : "Conv2d_3988"
          outputs : "Relu_7355"
          layer_params {
            num_features : "4 * growth_rate"
            eps : "1e-05"
            momentum : "0.1"
            affine : "True"
            track_running_stats : "True"
          }
        }
      }
      layer{
        act{
          name : "Relu_7355"
          layer_builder : "NNTorchLayer"
          layer_mode : RELU
          inputs : "BatchNorm2d_8569"
          outputs : "Conv2d_1564"
          layer_params {
            inplace : "False"
          }
        }
      }
      layer{
        conv{
          name : "Conv2d_1564"
          layer_builder : "NNTorchLayer"
          layer_mode : CONV2D
          inputs : "Relu_7355"
          layer_params {
            in_channels : "4 * growth_rate"
            out_channels : "growth_rate"
            kernel_size : "3"
            stride : "1"
            padding : "1"
            dilation : "1"
            groups : "1"
            bias : "False"
          }
        }
      }
    }

    seq_net {
      name : "after_block"
      layer{
        norm{
          name : "BatchNorm2d_7067"
          layer_builder : "NNTorchLayer"
          layer_mode : BATCHNORM2D
          outputs : "Relu_6697"
          layer_params {
            num_features : "num_planes"
            eps : "1e-05"
            momentum : "0.1"
            affine : "True"
            track_running_stats : "True"
          }
        }
      }
      layer{
        act{
          name : "Relu_6697"
          layer_builder : "NNTorchLayer"
          layer_mode : RELU
          inputs : "BatchNorm2d_7067"
          outputs : "AdaptiveMaxPool2d_4129"
          layer_params {
            inplace : "False"
          }
        }
      }
      layer{
        pool{
          name : "AdaptiveMaxPool2d_4129"
          layer_builder : "NNTorchLayer"
          layer_mode : ADAPTIVEMAXPOOL2D
          inputs : "Relu_6697"
          outputs : "Flatten_3502"
          layer_params {
            output_size : "1"
            output_size : "1"
            return_indices : "False"
          }
        }
      }
      layer{
        vulkan {
          name : "Flatten_3502"
          layer_builder : "NNTorchLayer"
          layer_mode : FLATTEN
          inputs : "AdaptiveMaxPool2d_4129"
          outputs : "Linear_9232"
          layer_params {
            start_dim : "1"
            end_dim : "-1"
          }
        }
      }
      layer{
        linear{
          name : "Linear_9232"
          layer_builder : "NNTorchLayer"
          layer_mode : LINEAR
          inputs : "Flatten_3502"
          layer_params {
            in_features : "num_planes"
            out_features : "num_classes"
            bias : "True"
          }
        }
      }
    }

    net_params {
      num_blocks : 6
      num_blocks : 12 
      num_blocks : 24 
      num_blocks : 16 

      growth_rate : 32

      reduction : 0.5

      num_classes : 10
    }
  }
  """
  # tmp_dir = '/tmp'
  with tempfile.TemporaryDirectory() as tmp_dir:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net_builder = NetBuilder(net_def)
    net = net_builder.build_net()
    net.write_net("{}/test_net.py".format(tmp_dir))
    with open("{}/test_net.py".format(tmp_dir)) as fin:
      for each_line in fin:
        print(each_line)
    spec = importlib.util.spec_from_file_location(
        "test_net", "{}/test_net.py".format(tmp_dir))
    test_net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_net)
    test_net_instance = test_net.custom_model()
    test_net_instance.to(device)
    test_input = torch.rand((4, 3, 224, 224)).to(device)
    # test_input = torch.rand((4, 3, 224, 224))
    out_tensor = test_net_instance(test_input)
    assert out_tensor.size() == torch.Size([4, 10])