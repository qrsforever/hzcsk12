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
        100000: {'en': 'success', 'cn': '成功'},
        100101: {'en': 'api parameter key is not found', 'cn': 'API参数错误: 非法Key'},
        100102: {'en': 'api parameter value is invalid', 'cn': 'API参数错误: 非法Value'},
        100103: {'en': 'api parameter json format error', 'cn': 'API参数错误: 非法Json格式'},
        100199: {'en': 'api parameter unknown exception', 'cn': 'API参数错误: Unknow异常'},

        100201: {'en': 'task is not found', 'cn': '任务启动错误: 请求的服务不存在或者服务未启动'},
        100202: {'en': 'task start fail', 'cn': '任务启动错误: 启动失败'},
        100203: {'en': 'task parameters is invalid', 'cn': '任务启动错误: 非法服务参数'},
        100204: {'en': 'task is running', 'cn': '任务启动错误: 服务已经启动,正在运行'},
        100205: {'en': 'task is not found or not running', 'cn': '任务停止错误: 请求的服务不存在或者未启动'},
        100206: {'en': 'task schema file is not found', 'cn': '模板请求错误: 模板文件不存在'},
        100207: {'en': 'task schema get fail', 'cn': '模板请求错误: 模板获取失败(内部错误)'},
        100208: {'en': 'task evaluate has no model.tar.gz', 'cn': '任务启动错误: 评估服务需要在完成训练后执行'},
        100209: {'en': 'task parameter has no key: input file', 'cn': '任务启动错误: 评估服务缺少输入文件'},
        100210: {'en': 'task parameter key is missing', 'cn': '任务启动错误: 缺少服务参数'},
        100299: {'en': 'task unknown exception', 'cn': '任务启动错误: Unkown异常'},

        100901: {'en': 'out of memory', 'cn': '常见错误: 内存溢出'},
        100902: {'en': 'not implemented yet', 'cn': '常见错误: 尚未实现'},
        999999: {'en': 'unknown error!', 'cn': '常见错误: 不可知错误'},
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
    print(msg)
    return msg
