#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file __init__.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-22 20:00

import sys, os, signal
import traceback
import psutil


def k12ai_except(func):
    def _wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            os.kill(1, signal.SIGUSR1)
    return _wrapper


def k12ai_kill(pid, parent=True):
    sys.stdout.flush()
    # os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    if parent:
        parent.kill()
    sys.exit(0) # TODO
