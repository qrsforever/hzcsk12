from __future__ import absolute_import, division, print_function

from vulkan.protos.layers import (activations_pb2, basefunc_pb2, conv_pb2,
                                  dropout_pb2, linear_pb2, pooling_pb2,
                                  vulkan_pb2, norm_pb2, padding_pb2)


class FuncNameDict():
  """ func to string map"""

  def __init__(self):
    # conv layers
    self.conv_func_dict = {
        conv_pb2.CONV1D: "nn.Conv1d",
        conv_pb2.CONV2D: "nn.Conv2d",
        conv_pb2.CONV3D: "nn.Conv3d",
        conv_pb2.TRANSCONV1D: "nn.ConvTranspose1d",
        conv_pb2.TRANSCONV2D: "nn.ConvTranspose2d",
        conv_pb2.TRANSCONV3D: "nn.ConvTranspose3d",
    }
    # pooling layers
    self.pool_func_dict = {
        # maxpooling methods
        pooling_pb2.MAXPOOL1D: "nn.MaxPool1d",
        pooling_pb2.MAXPOOL2D: "nn.MaxPool2d",
        pooling_pb2.MAXPOOL3D: "nn.MaxPool3d",

        # average pooling
        pooling_pb2.AVGPOOL1D: "nn.AvgPool1d",
        pooling_pb2.AVGPOOL2D: "nn.AvgPool2d",
        pooling_pb2.AVGPOOL3D: "nn.AvgPool3d",
        pooling_pb2.ADAPTIVEAVGPOOL1D: "nn.AdaptiveAvgPool1d",
        pooling_pb2.ADAPTIVEAVGPOOL2D: "nn.AdaptiveAvgPool2d",
        pooling_pb2.ADAPTIVEAVGPOOL3D: "nn.AdaptiveAvgPool3d",
        pooling_pb2.ADAPTIVEMAXPOOL1D: "nn.AdaptiveMaxPool1d",
        pooling_pb2.ADAPTIVEMAXPOOL2D: "nn.AdaptiveMaxPool2d",
        pooling_pb2.ADAPTIVEMAXPOOL3D: "nn.AdaptiveMaxPool3d",
        pooling_pb2.MAXUNPOOL1D: 'nn.MaxUnpool1d',
        pooling_pb2.MAXUNPOOL2D: 'nn.MaxUnpool2d',
        pooling_pb2.MAXUNPOOL3D: 'nn.MaxUnpool3d',
        pooling_pb2.LPPOOL1D: 'nn.LPPool1d',
        pooling_pb2.LPPOOL2D: 'nn.LPPool2d',
        pooling_pb2.FRACTIONALMAXPOOL2D: 'nn.FractionalMaxPool2d'
    }
    # dropout layers
    self.dropout_func_dict = {
        dropout_pb2.DROPOUT: "nn.Dropout",
        dropout_pb2.DROPOUT2D: "nn.Dropout2d",
        dropout_pb2.DROPOUT3D: "nn.Dropout3d",
        dropout_pb2.ALPHADROPOUT: "nn.AlphaDropout"
    }

    # activation layers
    self.act_func_dict = {
        activations_pb2.RELU: "nn.ReLU",
        activations_pb2.ELU: "nn.ELU",
        activations_pb2.HARDSHRINK: "nn.Hardshrink",
        activations_pb2.HARDTANH: "nn.Hardtanh",
        activations_pb2.LEAKYRELU: "nn.LeakyReLU",
        activations_pb2.LOGSIGMOID: "nn.LogSigmoid",
        activations_pb2.PRELU: "nn.PReLU",
        activations_pb2.RELU6: "nn.ReLU6",
        activations_pb2.RRELU: "nn.RReLU",
        activations_pb2.SELU: "nn.SELU",
        activations_pb2.CELU: "nn.CELU",
        activations_pb2.SIGMOID: "nn.Sigmoid",
        activations_pb2.SOFTPLUS: "nn.Softplus",
        activations_pb2.SOFTSHRINGK: "nn.Softshrink",
        activations_pb2.SOFTSIGN: "nn.Softsign",
        activations_pb2.TANH: "nn.Tanh",
        activations_pb2.TANHSHRINK: "nn.Tanhshrink",
        activations_pb2.THRESHHOLD: "nn.Threshold",
    }

    # linear layer
    self.linear_func_dict = {
        linear_pb2.LINEAR: "nn.Linear",
        linear_pb2.BILINEAR: "nn.Bilinear"
    }

    # basefunc layer
    self.basefunc_func_dict = {
        basefunc_pb2.CAT: "torch.cat",
        basefunc_pb2.ADD: "torch.add"
    }

    # vulkan layer
    self.vulkan_func_dict = {
        vulkan_pb2.RESHAPE: "vlc.Reshape",
        vulkan_pb2.FLATTEN: "vlc.Flatten"
    }

    # Norm layer
    self.norm_func_dict = {
        norm_pb2.BATCHNORM1D: "nn.BatchNorm1d",
        norm_pb2.BATCHNORM2D: "nn.BatchNorm2d",
        norm_pb2.BATCHNORM3D: "nn.BatchNorm3d",
        norm_pb2.GROUPNORM: "nn.GroupNorm",
        norm_pb2.INSTANCENORM1D: "nn.InstanceNorm1d",
        norm_pb2.INSTANCENORM2D: "nn.InstanceNorm2d",
        norm_pb2.INSTANCENORM3D: "nn.InstanceNorm3d",
        norm_pb2.LAYERNORM: "nn.LayerNorm",
        norm_pb2.LOCALRESPONSENORM: "nn.LocalResponseNorm"
    }

    # Padding layer
    self.padding_func_dict = {
        padding_pb2.REFLECTIONPAD1D: "nn.ReflectionPad1d",
        padding_pb2.REFLECTIONPAD2D: "nn.ReflectionPad2d",
        padding_pb2.REPLICATIONPAD1D: "nn.ReplicationPad1d",
        padding_pb2.REPLICATIONPAD2D: "nn.ReplicationPad2d",
        padding_pb2.REPLICATIONPAD3D: "nn.ReplicationPad3d",
        padding_pb2.ZEROPAD2D: "nn.ZeroPad2d",
        padding_pb2.CONSTANTPAD1D: "nn.ConstantPad1d",
        padding_pb2.CONSTANTPAD2D: "nn.ConstantPad2d",
        padding_pb2.CONSTANTPAD3D: "nn.ConstantPad3d"
    }
