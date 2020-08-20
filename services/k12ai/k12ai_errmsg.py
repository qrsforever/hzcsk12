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
        100000: {'en': 'success', 'cn': '成功'},
        100001: {'en': 'program staring', 'cn': '程序正在启动'},
        100002: {'en': 'program running', 'cn': '程序正在运行'},
        100003: {'en': 'program normal exit', 'cn': '程序正常结束'},
        100004: {'en': 'program stop manually', 'cn': '程序手动停止'},
        100005: {'en': 'program warning', 'cn': '程序运行警告'},
        
        100010: {'en': 'schema version is already lastest', 'cn': 'schema已是最新版本'},
        100011: {'en': 'payload is too large', 'cn': '警告: 负载数据太大'},


        100101: {'en': 'api parameter key is not found', 'cn': 'API参数错误: 非法Key'},
        100102: {'en': 'api parameter value is invalid', 'cn': 'API参数错误: 非法Value'},
        100103: {'en': 'api parameter json format error', 'cn': 'API参数错误: 非法Json格式'},
        100104: {'en': 'api parameter custom network is invalid', 'cn': 'API参数错误: 用户自定义网络错误'},
        100199: {'en': 'api parameter unknown exception', 'cn': 'API参数错误: Unknow异常'},

        100201: {'en': 'task service is not found', 'cn': '任务启动错误: 请求的服务不存在或者服务未启动'},
        100202: {'en': 'task start fail', 'cn': '任务启动错误: 启动失败'},
        100203: {'en': 'task start docker fail', 'cn': '任务启动错误: 容器启动失败'},
        100204: {'en': 'task is running', 'cn': '任务启动错误: 服务已经启动,正在运行'},
        100205: {'en': 'task is not found or not running', 'cn': '任务停止错误: 请求的服务不存在或者未启动'},
        100206: {'en': 'task schema file is not found', 'cn': '模板请求错误: 模板文件不存在'},
        100207: {'en': 'task schema get fail', 'cn': '模板请求错误: 模板获取失败(内部错误)'},
        100208: {'en': 'task evaluate has no model file', 'cn': '任务启动错误: 评估服务缺少模型文件'},
        100209: {'en': 'task evaluate has no config file', 'cn': '任务启动错误: 评估服务缺少配置文件'},
        100210: {'en': 'task starting too many', 'cn': '任务启动错误: 启动的任务太多'},
        100211: {'en': 'task model file is broken', 'cn': '任务启动错误: 模型文件已损坏'},
        100212: {'en': 'dataset file is missing', 'cn': '任务启动错误: 缺少数据集文件'},
        100213: {'en': 'task predict image file is missing', 'cn': '任务启动错误: 缺少预测文件'},
        100214: {'en': 'task resume config file is missing', 'cn': '任务启动错误: 缺少任务恢复配置文件'},

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
        100906: {'en': 'CUDA out of memory', 'cn': '常见错误: CUDA内存溢出'},
        100907: {'en': 'Vanishing gradient, Exploding gradient', 'cn': '常见错误: 梯度消失,梯度爆炸'},
        100908: {'en': 'docker image not found', 'cn': '常见错误: 任务镜像不存在'},
        100909: {'en': 'system command execute fail', 'cn': '常见错误: 系统命令执行错误'},
        100999: {'en': 'unkown error!', 'cn': '不可知错误'},

        999999: {'en': 'unknown error!', 'cn': '常见错误: 不可知错误'},
}


def gen_exc_info(errno=None):
    exc_type, exc_value, exc_tb = sys.exc_info()
    message = {
        'err_type': exc_type.__name__,
        'err_text': str(exc_value),
    }
    if isinstance(errno, int) and errno > 0:
        message['err_code'] = errno
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


def k12ai_error_message(code=100000, content=None, expand=None, exc=False):
    msg = {}
    msg['code'] = code
    txt = ERRORS.get(code, None)
    if txt:
        msg['message'] = txt

    if content:
        msg['data'] = content

    if expand:
        msg['expand'] = expand

    if exc:
        msg['expand'] = {}
        msg['expand']['excinfo'] = gen_exc_info()
    # print(msg)
    return msg
