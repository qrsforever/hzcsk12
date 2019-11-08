#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import GPUtil
import psutil


def get_gpu_status():
    """retrive gpu usage
  
  Returns:
    dict: dict contains the status of running GPU
  """

    gpu_info = GPUtil.getGPUs()[0]
    gpu_status = {
        "GPU_Type": "{}".format(gpu_info.name),
        "GPU_Mem_Free": gpu_info.memoryFree,
        "GPU_Mem_Total": gpu_info.memoryTotal,
        "GPU_Util": gpu_info.load * 100,
    }
    return gpu_status


def get_cpu_status():
    """retrieve cpu status
  """
    cpu_percent = psutil.cpu_percent()
    cpu_status = {"CPU_Util": cpu_percent}
    return cpu_status
