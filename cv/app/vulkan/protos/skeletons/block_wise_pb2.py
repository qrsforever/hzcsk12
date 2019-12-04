# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vulkan/protos/skeletons/block_wise.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from vulkan.protos.blocks import blocks_pb2 as vulkan_dot_protos_dot_blocks_dot_blocks__pb2
from vulkan.protos.skeletons import plain_pb2 as vulkan_dot_protos_dot_skeletons_dot_plain__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='vulkan/protos/skeletons/block_wise.proto',
  package='',
  syntax='proto2',
  serialized_pb=_b('\n(vulkan/protos/skeletons/block_wise.proto\x1a!vulkan/protos/blocks/blocks.proto\x1a#vulkan/protos/skeletons/plain.proto\"\x86\x01\n\x12\x42lockWiseNetParams\x12\x12\n\nnum_blocks\x18\x01 \x03(\r\x12\x0e\n\x06planes\x18\x02 \x03(\r\x12\x0f\n\x07strides\x18\x03 \x03(\r\x12\x13\n\x0bnum_classes\x18\x04 \x01(\r\x12\x13\n\x0bgrowth_rate\x18\x05 \x01(\r\x12\x11\n\treduction\x18\x06 \x01(\x02\"\x95\x01\n\x0c\x42lockWiseNet\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x1a\n\x07seq_net\x18\x02 \x03(\x0b\x32\t.PlainNet\x12\x16\n\x05\x62lock\x18\x03 \x01(\x0b\x32\x07.Blocks\x12\x1a\n\x08net_mode\x18\x04 \x01(\x0e\x32\x08.NetMode\x12\'\n\nnet_params\x18\x05 \x01(\x0b\x32\x13.BlockWiseNetParams*#\n\x07NetMode\x12\n\n\x06RESNET\x10\x00\x12\x0c\n\x08\x44\x45NSENET\x10\x01')
  ,
  dependencies=[vulkan_dot_protos_dot_blocks_dot_blocks__pb2.DESCRIPTOR,vulkan_dot_protos_dot_skeletons_dot_plain__pb2.DESCRIPTOR,])

_NETMODE = _descriptor.EnumDescriptor(
  name='NetMode',
  full_name='NetMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RESNET', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DENSENET', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=405,
  serialized_end=440,
)
_sym_db.RegisterEnumDescriptor(_NETMODE)

NetMode = enum_type_wrapper.EnumTypeWrapper(_NETMODE)
RESNET = 0
DENSENET = 1



_BLOCKWISENETPARAMS = _descriptor.Descriptor(
  name='BlockWiseNetParams',
  full_name='BlockWiseNetParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_blocks', full_name='BlockWiseNetParams.num_blocks', index=0,
      number=1, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='planes', full_name='BlockWiseNetParams.planes', index=1,
      number=2, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='strides', full_name='BlockWiseNetParams.strides', index=2,
      number=3, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='BlockWiseNetParams.num_classes', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='growth_rate', full_name='BlockWiseNetParams.growth_rate', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reduction', full_name='BlockWiseNetParams.reduction', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=117,
  serialized_end=251,
)


_BLOCKWISENET = _descriptor.Descriptor(
  name='BlockWiseNet',
  full_name='BlockWiseNet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='BlockWiseNet.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='seq_net', full_name='BlockWiseNet.seq_net', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='block', full_name='BlockWiseNet.block', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='net_mode', full_name='BlockWiseNet.net_mode', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='net_params', full_name='BlockWiseNet.net_params', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=254,
  serialized_end=403,
)

_BLOCKWISENET.fields_by_name['seq_net'].message_type = vulkan_dot_protos_dot_skeletons_dot_plain__pb2._PLAINNET
_BLOCKWISENET.fields_by_name['block'].message_type = vulkan_dot_protos_dot_blocks_dot_blocks__pb2._BLOCKS
_BLOCKWISENET.fields_by_name['net_mode'].enum_type = _NETMODE
_BLOCKWISENET.fields_by_name['net_params'].message_type = _BLOCKWISENETPARAMS
DESCRIPTOR.message_types_by_name['BlockWiseNetParams'] = _BLOCKWISENETPARAMS
DESCRIPTOR.message_types_by_name['BlockWiseNet'] = _BLOCKWISENET
DESCRIPTOR.enum_types_by_name['NetMode'] = _NETMODE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BlockWiseNetParams = _reflection.GeneratedProtocolMessageType('BlockWiseNetParams', (_message.Message,), dict(
  DESCRIPTOR = _BLOCKWISENETPARAMS,
  __module__ = 'vulkan.protos.skeletons.block_wise_pb2'
  # @@protoc_insertion_point(class_scope:BlockWiseNetParams)
  ))
_sym_db.RegisterMessage(BlockWiseNetParams)

BlockWiseNet = _reflection.GeneratedProtocolMessageType('BlockWiseNet', (_message.Message,), dict(
  DESCRIPTOR = _BLOCKWISENET,
  __module__ = 'vulkan.protos.skeletons.block_wise_pb2'
  # @@protoc_insertion_point(class_scope:BlockWiseNet)
  ))
_sym_db.RegisterMessage(BlockWiseNet)


# @@protoc_insertion_point(module_scope)