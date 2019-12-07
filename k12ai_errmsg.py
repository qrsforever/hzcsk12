#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_errmsg.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-22 14:42:09

import sys
import traceback

ERRORS = {
        100000: 'success',
        100100: '100100', # k12ai service api params parse error
        100101: 'api parameter key is not found',
        100102: 'api parameter value is invalid',
        100199: 'api parameter unkown exception',

        100200: 'task service success', # k12 services error (k12cv, k12nlp)
        100201: 'task service is not found',
        100202: 'task service start fails',
        100203: 'task service parameters is invalid',
        100299: 'task service unkown exception',

        100300: 'k12cv container success', # k12cv container inner process error
        100301: 'k12cv container is not found!',
        100302: 'k12cv container start fail',
        100303: 'k12cv container stop fail',
        100304: 'k12cv container is already running',
        100305: 'k12cv configuration error',
        100306: 'k12cv image type error',
        100307: 'k12cv tensor size error',
        100399: 'k12cv container unkown exception',

        100400: 'k12nlp container success', # k12nlp container inner process error
        100401: 'k12nlp container is not found',
        100402: 'k12nlp container start fail',
        100403: 'k12nlp container stop fail',
        100404: 'k12nlp container is already running',
        100405: 'k12nlp container process start configure error',
        100499: 'k12nlp continaer process unkown exception',

        100500: '100500',

        100600: '100600',

        100700: '100700',

        100800: '100800',

        100900: '100900', # common except error
        100901: 'common exception: out of memory',

        999999: 'Unkown error!',
}

def k12ai_error_message(code, message=None, detail=False, exc=False, ext_info=None):
    msg = {}
    msg['code'] = code
    txt = ERRORS.get(code, None)
    if txt:
        msg['descr'] = txt
    if message:
        msg['message'] = message
    if exc or ext_info:
        if ext_info:
            msg['detail'] = ext_info
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()
            msg['detail'] = {
                    'exc_type': exc_type.__name__,
                    'exc_text': str(exc_value)
                    } # noqa
            if exc_tb:
                msg['detail']['trackback'] = []
                tbs = traceback.extract_tb(exc_tb)
                for tb in tbs:
                    err = {
                            'filename': tb.filename,
                            'linenum': tb.lineno,
                            'funcname': tb.name,
                            'souce': tb.line
                            } # noqa
                    msg['detail']['trackback'].append(err)

    else:
        if detail:
            stack = traceback.extract_stack(limit=2)[0]
            msg['detail'] = {
                    'filename': stack[0],
                    'linenum': stack[1],
                    'funcname': stack[2]
                    } # noqa
    return msg
