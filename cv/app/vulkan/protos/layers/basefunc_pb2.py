# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vulkan/protos/layers/basefunc.proto

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




DESCRIPTOR = _descriptor.FileDescriptor(
  name='vulkan/protos/layers/basefunc.proto',
  package='',
  syntax='proto2',
  serialized_pb=_b('\n#vulkan/protos/layers/basefunc.proto\"\x1e\n\x0f\x42\x61seFunc_params\x12\x0b\n\x03\x64im\x18\x02 \x01(\t\"\x9c\x01\n\rBaseFuncLayer\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06inputs\x18\x02 \x03(\t\x12\x0f\n\x07outputs\x18\x03 \x03(\t\x12\x1d\n\nlayer_mode\x18\x04 \x01(\x0e\x32\t.BaseFunc\x12&\n\x0clayer_params\x18\x05 \x01(\x0b\x32\x10.BaseFunc_params\x12\x15\n\rlayer_builder\x18\x06 \x01(\t*\x1c\n\x08\x42\x61seFunc\x12\x07\n\x03\x43\x41T\x10\x01\x12\x07\n\x03\x41\x44\x44\x10\x02')
)

_BASEFUNC = _descriptor.EnumDescriptor(
  name='BaseFunc',
  full_name='BaseFunc',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CAT', index=0, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ADD', index=1, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=230,
  serialized_end=258,
)
_sym_db.RegisterEnumDescriptor(_BASEFUNC)

BaseFunc = enum_type_wrapper.EnumTypeWrapper(_BASEFUNC)
CAT = 1
ADD = 2



_BASEFUNC_PARAMS = _descriptor.Descriptor(
  name='BaseFunc_params',
  full_name='BaseFunc_params',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dim', full_name='BaseFunc_params.dim', index=0,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=39,
  serialized_end=69,
)


_BASEFUNCLAYER = _descriptor.Descriptor(
  name='BaseFuncLayer',
  full_name='BaseFuncLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='BaseFuncLayer.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='BaseFuncLayer.inputs', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='outputs', full_name='BaseFuncLayer.outputs', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_mode', full_name='BaseFuncLayer.layer_mode', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_params', full_name='BaseFuncLayer.layer_params', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_builder', full_name='BaseFuncLayer.layer_builder', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=72,
  serialized_end=228,
)

_BASEFUNCLAYER.fields_by_name['layer_mode'].enum_type = _BASEFUNC
_BASEFUNCLAYER.fields_by_name['layer_params'].message_type = _BASEFUNC_PARAMS
DESCRIPTOR.message_types_by_name['BaseFunc_params'] = _BASEFUNC_PARAMS
DESCRIPTOR.message_types_by_name['BaseFuncLayer'] = _BASEFUNCLAYER
DESCRIPTOR.enum_types_by_name['BaseFunc'] = _BASEFUNC
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BaseFunc_params = _reflection.GeneratedProtocolMessageType('BaseFunc_params', (_message.Message,), dict(
  DESCRIPTOR = _BASEFUNC_PARAMS,
  __module__ = 'vulkan.protos.layers.basefunc_pb2'
  # @@protoc_insertion_point(class_scope:BaseFunc_params)
  ))
_sym_db.RegisterMessage(BaseFunc_params)

BaseFuncLayer = _reflection.GeneratedProtocolMessageType('BaseFuncLayer', (_message.Message,), dict(
  DESCRIPTOR = _BASEFUNCLAYER,
  __module__ = 'vulkan.protos.layers.basefunc_pb2'
  # @@protoc_insertion_point(class_scope:BaseFuncLayer)
  ))
_sym_db.RegisterMessage(BaseFuncLayer)


# @@protoc_insertion_point(module_scope)
