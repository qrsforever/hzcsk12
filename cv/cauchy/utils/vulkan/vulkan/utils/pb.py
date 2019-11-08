from __future__ import absolute_import, division, print_function

from google.protobuf import text_format


def read_proto(pb_def, pb_templates, one_of_str):
  """read proto according to type
  
  Args:
    pb_def (str): pb definition in string
    pb_templates (protobuf descriptor): descriptor of particular protobuf
    one_of_str (str) : type of that pb
  """

  pb_def = text_format.Merge(pb_def, pb_templates)
  return eval("pb_def.{}".format(pb_def.WhichOneof(one_of_str)))