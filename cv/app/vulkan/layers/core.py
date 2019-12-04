from __future__ import absolute_import, division, print_function

import torch
from torch.nn import Conv2d
import numpy as np


class Reshape():
  """reshape layar for vulkan

  Args: 
    target_shape (tuple): a tupel describes the target shape 

  """

  def __init__(self, target_shape):
    self.target_shape = target_shape

  def _fix_unknown_dimension(self, input_shape, output_shape):
    """Finds and replaces a missing dimension in an output shape.
        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`
        # Arguments
            input_shape: original shape of array being reshaped
            output_shape: target shape of the array, with at most
                a single -1 which indicates a dimension that should be
                derived from the input shape.
        # Returns
            The new output shape with a `-1` replaced with its computed value.
        # Raises
            ValueError: if `input_shape` and `output_shape` do not match.
        """
    output_shape = list(output_shape)
    msg = 'total size of new array must be unchanged'

    known, unknown = 1, None
    for index, dim in enumerate(output_shape):
      if dim < 0:
        if unknown is None:
          unknown = index
        else:
          raise ValueError('Can only specify one unknown dimension.')
      else:
        known *= dim

    original = np.prod(input_shape, dtype=int)
    if unknown is not None:
      if known == 0 or original % known != 0:
        raise ValueError(msg)
      output_shape[unknown] = original // known
    elif original != known:
      raise ValueError(msg)

    return tuple(output_shape)

  def _compute_output_shape(self, input_tensor, out_shape):
    input_shape = list(input_tensor.shape)
    return (input_shape[0],) + self._fix_unknown_dimension(
        input_shape[1:], self.target_shape)

  def __call__(self, input_tensor):
    """reshape the input tensor into target shape"""

    output_shape = self._compute_output_shape(input_tensor, self.target_shape)
    return torch.reshape(input_tensor, output_shape)


class Flatten():
  """flatten tensors

  Args:
    start_dim (int): first dim to flatten 
    end_dim (int): last dim to flatten 
  """

  def __init__(self, start_dim=0, end_dim=-1):
    self.start_dim = start_dim
    self.end_dim = end_dim

  def __call__(self, input_tensor):
    return torch.flatten(
        input_tensor, start_dim=self.start_dim, end_dim=self.end_dim)
