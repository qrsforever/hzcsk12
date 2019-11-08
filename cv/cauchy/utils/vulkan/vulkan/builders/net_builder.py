from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from google.protobuf import text_format

from vulkan.builders.block_builder import Block_Builder

from ..protos import skeleton_pb2
from ..protos.skeletons import block_wise_pb2
from .layer_builder import LayerBuilder
from .net_templates import *
import os


class BaseNetBuilder():
  """ build layer code
  build layer code for plain_net TODO: densenet vgg resnet
  """

  def __init__(self, net_proto):
    # network definition
    self.proto = net_proto
    # layers in netdef
    self.layers = self.__parse_layers()

  def __parse_layers(self,):
    """parse layers in net_def """
    layer_generator = LayerBuilder()
    layers = []
    for layer_def in self.proto.layer:
      layers.append(layer_generator.create_layer(layer_def))
    return layers


class PlainNetBuilder(BaseNetBuilder):
  """build plain net with no specific design"""

  def __init__(self, net_proto):
    super(PlainNetBuilder, self).__init__(net_proto)

  def get_attr_clauses(self):
    """ generate attributes clauses"""
    attr_clauses = [layer.attributes_clause() for layer in self.layers]
    attr_clauses = [attr_clause + '\n' for attr_clause in attr_clauses]
    return "    ".join(attr_clauses)

  def get_forward_clauses(self):
    """generate forward clauses"""
    f_clauses = [layer.forward_clause() for layer in self.layers]
    # f_clauses[0] = self.layers[0].first_forward_clause()
    f_clauses.append("return {}_feat".format(self.layers[-1].proto.name))
    f_clauses = [f_clause + '\n' for f_clause in f_clauses]
    return "    ".join(f_clauses)

  def get_forward_clauses_as_seqnet(self, has_first_clause):
    """generate forward clauses regarding net as sequential type"""
    f_clauses = [layer.forward_clauses_for_seqnet() for layer in self.layers]
    f_clauses[0] = self.layers[0].forward_clauses_for_seqnet(has_first_clause)
    f_clauses = [f_clause + '\n' for f_clause in f_clauses]
    return "    ".join(f_clauses)

  def write_net(self, file_path):
    # print(net_templates.plain_net)
    # model_dir = os.path.dir
    with open(file_path, 'w') as fout:
      fout.write(
          PlainNetTemplate.format(self.proto.name.capitalize(),
                                  self.get_attr_clauses(),
                                  self.get_forward_clauses()))


class BaseBlockWiseNetBuilder():
  """build network follow the design of block wise network like resnet 
  or densenet"""

  def __init__(self, net_proto):
    self.proto = net_proto
    self.net_split = self._parse_net()
    # params for network
    self.net_params = self._parse_net_params()
    # block string
    self.block = self._parse_block()

  def _parse_net(self,):
    """ parse the net proto and split them into blocks and non-blocks"""
    net_split = {}

    # extract layer before and after blocks
    for seq_net in self.proto.seq_net:
      net_split[seq_net.name] = seq_net

    # extract net_block
    net_split['block'] = self.proto.block

    return net_split

  def _parse_seq_nets(self):
    """parse the seqnet to generate attributes clause and forward clauses"""
    seq_net_clauses = {}
    for (part, has_first_clause) in zip(["before_block", "after_block"],
                                        [True, False]):
      seq_net_clauses[part] = self._parse_seq_net(self.net_split[part],
                                                  has_first_clause)
    return seq_net_clauses

  def _parse_seq_net(self, seq_net_proto, has_first_clause):
    """pares single seq net and generate codes for particular part
    
    Args:
      seq_net_proto (pb_mesasge): a pb message contains the definition of the
      seqnet 
      has_first_clause (bool): whether this part contains first clause
    
    Returns:
      dict: a dict contains both attr clauses and forward clauses
    """

    attr_forward_clauses = {}

    # build net for particular part
    seq_net = PlainNetBuilder(seq_net_proto)

    #
    attr_forward_clauses['attr'] = seq_net.get_attr_clauses()
    attr_forward_clauses['fward'] = seq_net.get_forward_clauses_as_seqnet(
        has_first_clause)
    return attr_forward_clauses

  def _parse_block(self,):
    """parse the block proto message and generate attributes clauses
    
    Returns:
      str: string of clauses of attributes layer
    """

    block_proto = self.net_split['block']

    # build a block
    custom_block = Block_Builder(block_proto)

    return custom_block.build_block()

  def _parse_net_params(self,):
    net_params = self.proto.net_params
    kw_params = OrderedDict()
    for field in net_params.DESCRIPTOR.fields:
      if field.label == field.LABEL_REPEATED and len(
          getattr(net_params, field.name)) > 0:
        kw_params[field.name] = list(getattr(net_params, field.name))
      elif field.label == field.LABEL_OPTIONAL and net_params.HasField(
          field.name):
        kw_params[field.name] = getattr(net_params, field.name)
    return kw_params


class ResnetBuilder(BaseBlockWiseNetBuilder):
  """ build resnet """

  def __init__(self, net_proto):
    super(ResnetBuilder, self).__init__(net_proto)

  def _gen_net_arch(self):
    """ generate net definition acoording to the net proto"""
    seq_net_clauses = self._parse_seq_nets()
    return ResnetTemplate.net_arch().format(
        self.proto.name, 10, seq_net_clauses['before_block']['attr'],
        self._make_layer_generator(), seq_net_clauses['after_block']['attr'],
        seq_net_clauses['before_block']['fward'],
        self._forward_mklayer_clauses(),
        seq_net_clauses['after_block']['fward'])

  def _make_layer_generator(self,):
    inplanes_list = self.net_params['planes']
    strides_list = self.net_params['strides']
    layer_template = "self.layer{0} = self._make_layer(block, {1}, num_blocks[{2}], stride={3})\n"
    mk_layer_clauses = []
    assert len(inplanes_list) == len(strides_list), "parameter length not equal"
    for (i, (inplane, stride)) in enumerate(zip(inplanes_list, strides_list)):
      mk_layer_clauses.append(layer_template.format(i + 1, inplane, i, stride))
    return "    ".join(mk_layer_clauses)

  def _forward_mklayer_clauses(self,):
    """generate forward clauses for block groups
    
    Returns:
      str: a str describe fward clauses
    """

    layer_fward_templates = "out = self.layer{0}(out)\n"
    fward_clauses = []
    for i in range(len(self.net_params['planes'])):
      fward_clauses.append(layer_fward_templates.format(i + 1))
    return "    ".join(fward_clauses)

  def write_net(self, file_path):
    num_blocks = '[{}]'.format(','.join(
        [str(num) for num in self.net_params['num_blocks']]))
    num_classes = str(self.net_params['num_classes'])
    with open(file_path, 'w') as fout:
      fout.write(ResnetTemplate.resnet_bundle().format(
          self.block.gen_block(), self._gen_net_arch(), self.proto.name,
          self.block.proto.name, num_blocks, num_classes))


class DensenetBuilder(BaseBlockWiseNetBuilder):
  """build densenet"""

  def __init__(self, net_proto):
    super(DensenetBuilder, self).__init__(net_proto)

  def _gen_custom_model(self):
    custom_model_template = "def custom_model():\n" \
                            "  return {0}({1}, {2}, growth_rate={3}," \
                            "reduction={4}, num_classes={5})"
    num_blocks = '[{}]'.format(','.join(
        [str(num) for num in self.net_params['num_blocks']]))
    return custom_model_template.format(self.proto.name, self.proto.block.name,
                                        num_blocks,
                                        str(self.net_params['growth_rate']),
                                        str(self.net_params['reduction']),
                                        str(self.net_params['num_classes']))

  def write_net(self, file_path):
    seq_net_clauses = self._parse_seq_nets()
    with open(file_path, 'w') as fout:
      fout.write(DensenetTemplate.densenet_bundle().format(
          self.block.gen_block(), self.proto.name,
          seq_net_clauses['before_block']['attr'],
          seq_net_clauses['after_block']['attr'],
          seq_net_clauses['before_block']['fward'],
          seq_net_clauses['after_block']['fward'], self._gen_custom_model()))


class BlockWiseNetBuilder():
  """build block wise network"""

  def __init__(self,):
    self.build_dict = {
        block_wise_pb2.RESNET: ResnetBuilder,
        block_wise_pb2.DENSENET: DensenetBuilder,
    }

  def build_net(self, net_proto):
    try:
      return self.build_dict[net_proto.net_mode](net_proto)
    except KeyError:
      print("network type not supported")
      raise


class NetBuilder():
  """ build net with net def"""

  def __init__(self, net_def):
    self.net_proto, self.net_type = self._parse_net_proto(net_def)

  def _parse_net_proto(self, net_def):
    net_proto = text_format.Merge(net_def, skeleton_pb2.skeletons())
    net_type = net_proto.WhichOneof("skeleton_oneof")
    net_proto = eval("net_proto.{}".format(net_type))
    return net_proto, net_type

  def build_net(self):
    if self.net_type == "plain_net":
      return PlainNetBuilder(self.net_proto)
    elif self.net_type == "block_wise_net":
      block_wise_buidler = BlockWiseNetBuilder()
      return block_wise_buidler.build_net(self.net_proto)
    else:
      raise NotImplementedError
