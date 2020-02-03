#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_logger.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-03 11:07

import logging
from logging import handlers


_LEVELS_ = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}
_LOG_LEVEL_ = 'info'
_LOG_FILENAME_ = 'k12ai'


def k12ai_set_loglevel(level):
    global _LOG_LEVEL_
    _LOG_LEVEL_ = level


def k12ai_set_logfile(filename):
    global _LOG_FILENAME_
    _LOG_FILENAME_ = filename


class Logger(object):

    logger = None

    @staticmethod
    def init(
            filename, level='info', when='D', backCount=3,
            fmt='%(asctime)s - %(filename)s - %(funcName)s:%(lineno)d - %(levelname)s: %(message)s'):  # noqa
        # pathname, filename
        Logger.logger = logging.getLogger(filename)
        Logger.logger.setLevel(_LEVELS_.get(level))

        console = logging.StreamHandler()
        logfile = handlers.TimedRotatingFileHandler(
                filename=filename,
                when=when,
                backupCount=backCount,
                encoding='utf-8')

        format = logging.Formatter(fmt)
        console.setFormatter(format)
        logfile.setFormatter(format)
        Logger.logger.addHandler(console)
        Logger.logger.addHandler(logfile)

    @staticmethod
    def check_logger():
        if Logger.logger is None:
            Logger.init(_LOG_FILENAME_, _LOG_LEVEL_)

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
