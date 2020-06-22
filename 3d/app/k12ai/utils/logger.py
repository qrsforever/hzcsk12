#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file logger.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-11 19:09


import logging


_LEVELS_ = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}
_LOG_LEVEL_ = 'info'


def k12ai_set_loglevel(level):
    global _LOG_LEVEL_
    _LOG_LEVEL_ = level


class Logger(object):

    logger = None

    @staticmethod
    def init(
            filename, level='info', when='D', backCount=3,
            fmt='%(asctime)s - %(levelname)s: %(message)s'):  # noqa
        # pathname, filename
        Logger.logger = logging.getLogger(filename)
        Logger.logger.setLevel(_LEVELS_.get(level))

        console = logging.StreamHandler()

        format = logging.Formatter(fmt)
        console.setFormatter(format)
        Logger.logger.addHandler(console)

    @staticmethod
    def check_logger():
        if Logger.logger is None:
            Logger.init('k12ml', level=_LOG_LEVEL_)

    @staticmethod
    def debug(message):
        Logger.check_logger()
        Logger.logger.debug(message)

    @staticmethod
    def info(message):
        Logger.check_logger()
        Logger.logger.info(message)

    @staticmethod
    def warning(message):
        Logger.check_logger()
        Logger.logger.waring(message)

    @staticmethod
    def error(message):
        Logger.check_logger()
        Logger.logger.error(message)

    @staticmethod
    def critical(message):
        Logger.check_logger()
        Logger.logger.critical(message)
