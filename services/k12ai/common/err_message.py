#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file err_message.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-01 23:56

import sys
import traceback


def k12ai_except_message():
    exc_type, exc_value, exc_tb = sys.exc_info()
    message = {
        'err_type': exc_type.__name__,
        'err_text': str(exc_value)
    }
    message['trackback'] = []
    tbs = traceback.extract_tb(exc_tb)
    for tb in tbs:
        err = {
            'filename': tb.filename,
            'linenum': tb.lineno,
            'funcname': tb.name,
            'source': tb.line
        }
        message['trackback'].append(err)
    return message
