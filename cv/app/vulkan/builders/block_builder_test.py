from __future__ import absolute_import, division, print_function

from google.protobuf import text_format

from vulkan.builders.block_builder import Block_Builder
from vulkan.protos.blocks import blocks_pb2


def test_res_block_buidler():
  block_def = r"""
    name : "resblock"
    block_mode : RES_BOTTLENECK 
    layer {
      conv {
        name : "conv1"
        inputs : "layer1"
        outputs : "layer3"
        layer_params {
          in_channels : "in_planes" 
          out_channels : "planes" 
          kernel_size : "stride"
        }
        layer_builder : "NNTorchLayer"
      }
    }

    layer {
      norm {
        name : "bn1"
        inputs : "layer1"
        outputs : "layer2"
        layer_params {
          num_features : "in_planes" 
        }
        layer_builder : "NNTorchLayer"
      }
    }
    
  """
  # parse the block content
  block_proto = blocks_pb2.Blocks()
  text_format.Merge(block_def, block_proto)

  # build a block
  cus_block = Block_Builder(block_proto)
  print(cus_block.build_block().gen_block())


def test_dense_block_builder():
  block_def = r"""
  name : "DenseBlock"
  block_mode : DENSE_BOTTLENECK
  layer{
  norm{
    name : "BatchNorm2d_8311"
    layer_builder : "NNTorchLayer"
    layer_mode : BATCHNORM2D
    outputs : "Conv2d_8097"
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
  conv{
    name : "Conv2d_8097"
    layer_builder : "NNTorchLayer"
    layer_mode : CONV2D
    inputs : "BatchNorm2d_8311"
    outputs : "BatchNorm2d_5545"
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
    name : "BatchNorm2d_5545"
    layer_builder : "NNTorchLayer"
    layer_mode : BATCHNORM2D
    inputs : "Conv2d_8097"
    outputs : "Conv2d_7063"
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
  conv{
    name : "Conv2d_7063"
    layer_builder : "NNTorchLayer"
    layer_mode : CONV2D
    inputs : "BatchNorm2d_5545"
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
}"""
  block_proto = blocks_pb2.Blocks()
  text_format.Merge(block_def, block_proto)

  # build a block
  cus_block = Block_Builder(block_proto)
  print(cus_block.build_block().gen_block())