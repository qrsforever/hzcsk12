from __future__ import absolute_import, division, print_function
from vulkan.protos.layers import basefunc_pb2

from collections import OrderedDict

from ._func_dict import FuncNameDict


class BasePytorchLayer():
  """build torch layer according to the message in proto

  Args:
  layer_proto (layer.proto): a proto desribes layer's name, mode, and params

  """

  def __init__(self, layer_proto, func_dict):
    self.proto = layer_proto
    self.params = self.extract_param(layer_proto.layer_params)
    self.func_dict = func_dict

  def _gen_non_shape_tuple(self, tuple_values):
    """generate string for non shape tuple"""
    if len(tuple_values) == 1:
      return "{}".format(tuple_values[0])
    else:
      return "({})".format(",".join(tuple_values))

  def _gen_shape_tuple(self, tuple_values):
    """ generatge string for shape tuple"""
    if len(tuple_values) == 1:
      return "({},)".format(tuple_values[0])
    else:
      return "({})".format(",".join(tuple_values))

  def tuple_generator(self, tuple_values, field_name):
    """generate texts of tuple for repeated fields
    
    Args:
      tuple_values ```int``` or ```tuple```: an integer or list of values
    
    Returns:
      str : a string represents the tuple or int
    """
    tuple_values = [str(v) for v in tuple_values]
    if field_name in ["shape", "target_shape"]:
      return self._gen_shape_tuple(tuple_values)
    else:
      return self._gen_non_shape_tuple(tuple_values)

  def extract_param(self, layer_params):
    """extract fields that are setted in proto"""
    key_params = OrderedDict()
    for field in layer_params.DESCRIPTOR.fields:
      if field.label == field.LABEL_OPTIONAL and layer_params.HasField(
          field.name):
        key_params[field.name] = getattr(layer_params, field.name)
      elif field.label == field.LABEL_REPEATED and len(
          getattr(layer_params, field.name)) > 0:
        key_params[field.name] = self.tuple_generator(
            getattr(layer_params, field.name), field.name)
      else:
        pass
    params = ', '.join(["{}={}".format(k, v) for k, v in key_params.items()])
    return params

  def _parse_input_name(self, input_str):
    if input_str == 'x':
      return 'x'
    else:
      return '{}_feat'.format(input_str)

  def _get_inputs(self, inputs):
    """ generate inputs string
    Args:
      inputs (proto.inputs or proto.outputs): inputs or outputs field of 
      proto
    Return:
      a ```string``` describes the inputs or outputs
    """
    if len(inputs) == 1:
      return self._parse_input_name(inputs[0])
    else:
      return ', '.join(
          [self._parse_input_name(each_input) for each_input in inputs])


class NNTorchLayer(BasePytorchLayer):
  """ construct torch.nn layers"""

  def __init__(self, layer_proto, func_dict):
    super(NNTorchLayer, self).__init__(layer_proto, func_dict)

  def generate_python_clause(self, layer_name, func_name):
    """ generate python clause
    """
    return "self.{} = {}({})".format(layer_name, func_name, self.params)

  def attributes_clause(self):
    return self.generate_python_clause(self.proto.name,
                                       self.func_dict[self.proto.layer_mode])

  def forward_clause(self):
    return "{0}_feat = self.{0}({1})".format(
        self.proto.name, self._get_inputs(self.proto.inputs))

  def forward_clauses_for_seqnet(self, has_first_clause=False):
    """generate forward clauses for seqnet"""
    if has_first_clause:
      input_str = 'x'
    else:
      input_str = 'out'
    output_str = 'out'
    return "{0} = self.{1}({2})".format(output_str, self.proto.name, input_str)

  def first_forward_clause(self):
    return "{0}_feat = self.{0}(x)".format(self.proto.name)

  def _module_clause(self,):
    """generate a clause repenseting the module
    
    Returns:
      string: a string describes the op
    """
    return "{}({})".format(self.func_dict[self.proto.layer_mode], self.params)

  def get_func_module(self,):
    """return the object of the module corresponding to the string
    
    Returns:
      torch.nn.module:  a ```torch.nn.module``` of after eval the string
    """
    return eval(self._module_clause())


class TorchFuncLayer(BasePytorchLayer):
  """create layer for base function
  Different from Pytorch Layer, base function layers has no need to generate 
  python clause for class attributes
  """

  def __init__(self, layer_proto, func_dict):
    super(TorchFuncLayer, self).__init__(layer_proto, func_dict)

  def attributes_clause(self):
    return ""

  def forward_clause(self):
    if self.proto.layer_mode == basefunc_pb2.ADD:
      forward_clause = "{0}_feat = {1}({2})".format(
          self.proto.name, self.func_dict[self.proto.layer_mode],
          self._get_inputs(self.proto.inputs))
    else:
      forward_clause = "{0}_feat = {1}(({2}), {3})".format(
          self.proto.name, self.func_dict[self.proto.layer_mode],
          self._get_inputs(self.proto.inputs), self.params)
    return forward_clause

  def first_forward_clause(self):
    return "{}_feat = {}(x, {})".format(
        self.proto.name, self.func_dict[self.proto.layer_mode], self.params)


class LayerBuilder():
  """generate layers for Pytorch or TensorFlow

  description about the class to use shoud be contained in the proto
  """

  def __init__(self,):
    self.func_dicts = FuncNameDict()
    self.layer_builders = {
        "NNTorchLayer": NNTorchLayer,
        "TorchFuncLayer": TorchFuncLayer
    }

  def get_fileds(self, layer_proto):
    """return corresponding layer according to the field name"""
    # return corresponding field
    layer_type = layer_proto.WhichOneof("layer_oneof")
    return eval("layer_proto.{}".format(layer_type))

  def get_func_dict(self, layer_proto):
    """ return corresponding func dict according to layer type"""
    try:
      layer_type = layer_proto.WhichOneof("layer_oneof")
      # print(layer_type)
      return eval("self.func_dicts.{}_func_dict".format(layer_type))
    except KeyError:
      print("layer not supported")
      raise

  def create_layer(self, layer_proto):
    """create layer according to the info recored in the proto

    """
    # extract layer info according to the layer type
    layer_proto_def = self.get_fileds(layer_proto)

    # build layer with the info
    try:
      func_dict = self.get_func_dict(layer_proto)
      return self.layer_builders[layer_proto_def.layer_builder](layer_proto_def,
                                                                func_dict)
    except KeyError:
      print("layer not supported")
      raise
