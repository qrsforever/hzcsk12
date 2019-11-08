#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from cauchy.utils.tools.logger import Logger as Log


def select_samples(res_list, num_per_cls):
    """randomly select items in results
  
  Args:
    res_list (list): list of results
    num_per_cls (int): item to select for each class 
  
  Returns:
    list: all randomly selected items 
  """
    res_dict = {}
    # group results by class
    for i, _res in enumerate(res_list):
        if _res in res_dict.keys():
            res_dict[_res].append(i)
        else:
            res_dict[_res] = [i]
    select_samples = []
    for cls_num, indices in res_dict.items():
        select_samples.extend(partial_random_list(indices, num_per_cls))
    return select_samples


def partial_list_items(orig_list, selected_indices):
    """fetch items according to the indices
  
  Args:
    orig_list (list): original list
    selected_indices (list): list of selected indices
  
  Returns:
    list: partial list
  """
    return [orig_list[i] for i in selected_indices]


def partial_random_list(orig_list, selected_nums):
    """random selected partial of list
  
  Args:
    orig_list (list): original list
    selected_nums (int): number of items to select
  
  Returns:
    list: partial of original list
  """

    list_sum = len(orig_list)
    # assert selected_nums <= list_sum
    try:
        assert selected_nums <= list_sum, "original images are not enough"
    except AssertionError:
        Log.error("original images are not enough")
        raise
    selected_indices = np.random.choice(list_sum, selected_nums)
    return partial_list_items(orig_list, selected_indices)
