#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_errmsg.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-22 14:42:09

import sys
import traceback

SUCCESS = 100000
FAILURE = -1

ERRORS = {
        100000: 'success',
        100100: '100100', # k12ai service api params parse error
        100101: 'api parameter key is not found',
        100102: 'api parameter value is invalid',
        100103: 'api parameter json format error',
        100199: 'api parameter unknown exception',

        100200: 'success', # k12 services error (k12cv, k12nlp)
        100201: 'task service is not found',
        100202: 'task service start fails',
        100203: 'task service parameters is invalid',
        100204: 'task service is running',
        100205: 'task service is not found or not running',
        100206: 'task service schema file is not found',
        100207: 'task service schema get fail',
        100299: 'task service unknown exception',

        100300: 'success', # k12cv container inner process error
        100301: 'container is not found!',
        100302: 'container start fail',
        100303: 'container stop fail',
        100304: 'container is running',
        100305: 'configuration error',
        100306: 'image type error',
        100307: 'tensor size error',
        100399: 'container unknown exception',

        100400: 'success', # k12nlp container inner process error
        100401: 'k12nlp container is not found',
        100402: 'k12nlp container start fail',
        100403: 'k12nlp container stop fail',
        100404: 'k12nlp container is running',
        100405: 'k12nlp container process start configure error',
        100499: 'k12nlp continaer process unknown exception',

        100500: '100500',

        100600: '100600',

        100700: '100700',

        100800: '100800',

        100900: '100900', # common except error
        100901: 'common exception: out of memory',
        100902: 'common exception: not implemented yet',

        999999: 'unknown error!',
}


def k12ai_error_message(code=100000, data=None, detail=False, exc=False, ext_info=None):
    msg = {}
    msg['code'] = code
    txt = ERRORS.get(code, None)
    if txt:
        msg['message'] = txt
    if data:
        msg['data'] = data
    if exc or ext_info:
        if ext_info:
            msg['detail'] = ext_info
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()
            msg['detail'] = {
                    'err_type': exc_type.__name__,
                    'err_text': str(exc_value)
                    } # noqa
            if exc_tb:
                msg['detail']['trackback'] = []
                tbs = traceback.extract_tb(exc_tb)
                for tb in tbs:
                    err = {
                            'filename': tb.filename,
                            'linenum': tb.lineno,
                            'funcname': tb.name,
                            'source': tb.line
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
