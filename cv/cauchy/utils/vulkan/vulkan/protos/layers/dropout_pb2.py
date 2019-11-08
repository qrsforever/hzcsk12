# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vulkan/protos/layers/dropout.proto

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
  name='vulkan/protos/layers/dropout.proto',
  package='',
  syntax='proto2',
  serialized_pb=_b('\n\"vulkan/protos/layers/dropout.proto\",\n\x0e\x64ropout_params\x12\x0f\n\x07inplace\x18\x01 \x01(\t\x12\t\n\x01p\x18\x02 \x01(\t\"\x9d\x01\n\x0c\x44ropoutLayer\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06inputs\x18\x02 \x03(\t\x12\x0f\n\x07outputs\x18\x03 \x03(\t\x12 \n\nlayer_mode\x18\x04 \x01(\x0e\x32\x0c.DropoutMode\x12%\n\x0clayer_params\x18\x05 \x01(\x0b\x32\x0f.dropout_params\x12\x15\n\rlayer_builder\x18\x06 \x01(\t*J\n\x0b\x44ropoutMode\x12\x0b\n\x07\x44ROPOUT\x10\x00\x12\r\n\tDROPOUT2D\x10\x01\x12\r\n\tDROPOUT3D\x10\x02\x12\x10\n\x0c\x41LPHADROPOUT\x10\x03')
)

_DROPOUTMODE = _descriptor.EnumDescriptor(
  name='DropoutMode',
  full_name='DropoutMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DROPOUT', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DROPOUT2D', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DROPOUT3D', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALPHADROPOUT', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=244,
  serialized_end=318,
)
_sym_db.RegisterEnumDescriptor(_DROPOUTMODE)

DropoutMode = enum_type_wrapper.EnumTypeWrapper(_DROPOUTMODE)
DROPOUT = 0
DROPOUT2D = 1
DROPOUT3D = 2
ALPHADROPOUT = 3



_DROPOUT_PARAMS = _descriptor.Descriptor(
  name='dropout_params',
  full_name='dropout_params',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='inplace', full_name='dropout_params.inplace', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='p', full_name='dropout_params.p', index=1,
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
  serialized_start=38,
  serialized_end=82,
)


_DROPOUTLAYER = _descriptor.Descriptor(
  name='DropoutLayer',
  full_name='DropoutLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='DropoutLayer.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='DropoutLayer.inputs', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='outputs', full_name='DropoutLayer.outputs', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_mode', full_name='DropoutLayer.layer_mode', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_params', full_name='DropoutLayer.layer_params', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_builder', full_name='DropoutLayer.layer_builder', index=5,
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
  serialized_start=85,
  serialized_end=242,
)

_DROPOUTLAYER.fields_by_name['layer_mode'].enum_type = _DROPOUTMODE
_DROPOUTLAYER.fields_by_name['layer_params'].message_type = _DROPOUT_PARAMS
DESCRIPTOR.message_types_by_name['dropout_params'] = _DROPOUT_PARAMS
DESCRIPTOR.message_types_by_name['DropoutLayer'] = _DROPOUTLAYER
DESCRIPTOR.enum_types_by_name['DropoutMode'] = _DROPOUTMODE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

dropout_params = _reflection.GeneratedProtocolMessageType('dropout_params', (_message.Message,), dict(
  DESCRIPTOR = _DROPOUT_PARAMS,
  __module__ = 'vulkan.protos.layers.dropout_pb2'
  # @@protoc_insertion_point(class_scope:dropout_params)
  ))
_sym_db.RegisterMessage(dropout_params)

DropoutLayer = _reflection.GeneratedProtocolMessageType('DropoutLayer', (_message.Message,), dict(
  DESCRIPTOR = _DROPOUTLAYER,
  __module__ = 'vulkan.protos.layers.dropout_pb2'
  # @@protoc_insertion_point(class_scope:DropoutLayer)
  ))
_sym_db.RegisterMessage(DropoutLayer)


# @@protoc_insertion_point(module_scope)
