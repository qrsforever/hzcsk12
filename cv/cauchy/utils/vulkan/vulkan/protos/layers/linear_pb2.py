# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vulkan/protos/layers/linear.proto

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
  name='vulkan/protos/layers/linear.proto',
  package='',
  syntax='proto2',
  serialized_pb=_b('\n!vulkan/protos/layers/linear.proto\"w\n\x10linear_parameter\x12\x13\n\x0bin_features\x18\x01 \x01(\t\x12\x14\n\x0cout_features\x18\x02 \x01(\t\x12\x0c\n\x04\x62ias\x18\x03 \x01(\t\x12\x14\n\x0cin1_features\x18\x04 \x01(\t\x12\x14\n\x0cin2_features\x18\x05 \x01(\t\"\x9d\x01\n\x0bLinearLayer\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06inputs\x18\x02 \x03(\t\x12\x0f\n\x07outputs\x18\x03 \x03(\t\x12\x1f\n\nlayer_mode\x18\x04 \x01(\x0e\x32\x0b.LinearMode\x12\'\n\x0clayer_params\x18\x05 \x01(\x0b\x32\x11.linear_parameter\x12\x15\n\rlayer_builder\x18\x06 \x01(\t*&\n\nLinearMode\x12\n\n\x06LINEAR\x10\x00\x12\x0c\n\x08\x42ILINEAR\x10\x01')
)

_LINEARMODE = _descriptor.EnumDescriptor(
  name='LinearMode',
  full_name='LinearMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LINEAR', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BILINEAR', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=318,
  serialized_end=356,
)
_sym_db.RegisterEnumDescriptor(_LINEARMODE)

LinearMode = enum_type_wrapper.EnumTypeWrapper(_LINEARMODE)
LINEAR = 0
BILINEAR = 1



_LINEAR_PARAMETER = _descriptor.Descriptor(
  name='linear_parameter',
  full_name='linear_parameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='in_features', full_name='linear_parameter.in_features', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='out_features', full_name='linear_parameter.out_features', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias', full_name='linear_parameter.bias', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='in1_features', full_name='linear_parameter.in1_features', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='in2_features', full_name='linear_parameter.in2_features', index=4,
      number=5, type=9, cpp_type=9, label=1,
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
  serialized_start=37,
  serialized_end=156,
)


_LINEARLAYER = _descriptor.Descriptor(
  name='LinearLayer',
  full_name='LinearLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='LinearLayer.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='LinearLayer.inputs', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='outputs', full_name='LinearLayer.outputs', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_mode', full_name='LinearLayer.layer_mode', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_params', full_name='LinearLayer.layer_params', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_builder', full_name='LinearLayer.layer_builder', index=5,
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
  serialized_start=159,
  serialized_end=316,
)

_LINEARLAYER.fields_by_name['layer_mode'].enum_type = _LINEARMODE
_LINEARLAYER.fields_by_name['layer_params'].message_type = _LINEAR_PARAMETER
DESCRIPTOR.message_types_by_name['linear_parameter'] = _LINEAR_PARAMETER
DESCRIPTOR.message_types_by_name['LinearLayer'] = _LINEARLAYER
DESCRIPTOR.enum_types_by_name['LinearMode'] = _LINEARMODE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

linear_parameter = _reflection.GeneratedProtocolMessageType('linear_parameter', (_message.Message,), dict(
  DESCRIPTOR = _LINEAR_PARAMETER,
  __module__ = 'vulkan.protos.layers.linear_pb2'
  # @@protoc_insertion_point(class_scope:linear_parameter)
  ))
_sym_db.RegisterMessage(linear_parameter)

LinearLayer = _reflection.GeneratedProtocolMessageType('LinearLayer', (_message.Message,), dict(
  DESCRIPTOR = _LINEARLAYER,
  __module__ = 'vulkan.protos.layers.linear_pb2'
  # @@protoc_insertion_point(class_scope:LinearLayer)
  ))
_sym_db.RegisterMessage(LinearLayer)


# @@protoc_insertion_point(module_scope)
