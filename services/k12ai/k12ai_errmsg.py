#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_errmsg.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-22 14:42:09

import traceback

ERRORS = {
        100000: {'en': 'success', 'cn': '成功'},
        100101: {'en': 'api parameter key is not found', 'cn': 'API参数错误: 非法Key'},
        100102: {'en': 'api parameter value is invalid', 'cn': 'API参数错误: 非法Value'},
        100103: {'en': 'api parameter json format error', 'cn': 'API参数错误: 非法Json格式'},
        100199: {'en': 'api parameter unknown exception', 'cn': 'API参数错误: Unknow异常'},

        100201: {'en': 'task service is not found', 'cn': '任务启动错误: 请求的服务不存在或者服务未启动'},
        100202: {'en': 'task start fail', 'cn': '任务启动错误: 启动失败'},
        100203: {'en': 'task start docker fail', 'cn': '任务启动错误: 容器启动失败'},
        100204: {'en': 'task is running', 'cn': '任务启动错误: 服务已经启动,正在运行'},
        100205: {'en': 'task is not found or not running', 'cn': '任务停止错误: 请求的服务不存在或者未启动'},
        100206: {'en': 'task schema file is not found', 'cn': '模板请求错误: 模板文件不存在'},
        100207: {'en': 'task schema get fail', 'cn': '模板请求错误: 模板获取失败(内部错误)'},
        100208: {'en': 'task evaluate has no model file', 'cn': '任务启动错误: 评估服务需要在完成训练后执行'},
        100209: {'en': 'task evaluate has no config file', 'cn': '任务启动错误: 评估服务缺少配置文件'},

        100231: {'en': 'task parameter is invalid', 'cn': '任务启动错误: 非法服务参数'},
        100232: {'en': 'task parameter has no key: input file', 'cn': '任务启动错误: 评估服务缺少输入文件'},
        100233: {'en': 'task parameter key is missing', 'cn': '任务启动错误: 缺少服务参数'},
        100299: {'en': 'task unknown exception', 'cn': '任务启动错误: Unkown异常'},

        100302: {'en': 'cv model name is not valid', 'cn': '框架运行错误: CV框架非法模型'},
        100303: {'en': 'cv optimizer method is not valid', 'cn': '框架运行错误: CV框架非法优化方法'},
        100304: {'en': 'cv pad mode is not valid', 'cn': '框架运行错误: CV框架非法填充模式'},
        100305: {'en': 'cv anchor method is not valid', 'cn': '框架运行错误: CV框架非法Anchor方式'},
        100306: {'en': 'cv image format type is not valid', 'cn': '框架运行错误: CV框架非法图片类型'},
        100307: {'en': 'cv tensor size error', 'cn': '框架运行错误: CV框架Tensor大小不合理'},

        100901: {'en': 'out of memory', 'cn': '常见错误: 内存溢出'},
        100902: {'en': 'not implemented yet', 'cn': '常见错误: 尚未实现'},
        100903: {'en': 'configuration error', 'cn': '常见错误: 参数配置错误'},
        100904: {'en': 'missing key configuration', 'cn': '常见错误: 配置中缺少参数'},
        100905: {'en': 'file is not found', 'cn': '常见错误: 文件不存在'},
        999999: {'en': 'unknown error!', 'cn': '常见错误: 不可知错误'},
}


def gen_exc_info():
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


def k12ai_error_message(code=100000, data=None, stack=False, exc=False, exc_info=None):
    msg = {}
    msg['code'] = code
    txt = ERRORS.get(code, None)
    if txt:
        msg['message'] = txt
    if data:
        msg['data'] = data
    if exc:
        msg['detail'] = gen_exc_info()
    else:
        if exc_info:
            msg['detail'] = exc_info
        else:
            if stack:
                stack = traceback.extract_stack(limit=2)[0]
                msg['detail'] = {
                    'filename': stack[0],
                    'linenum': stack[1],
                    'funcname': stack[2]
                }
    print(msg)
    return msg
