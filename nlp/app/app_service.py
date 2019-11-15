#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-15 16:38:16

import argparse
import zerorpc
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.debug('start zerorpc server on port')

class AppServiceRPC(object):

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--port',
            default=None,
            type=int,
            dest='port',
            help="port to run app service")
    parser.add_argument(
            '--loglevel',
            default=logging.WARNING,
            type=int,
            dest='loglevel',
            help="loglevel to run app service")
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    logger.debug('start zerorpc server on port: %d', args.port)

    app = zerorpc.Server(AppServiceRPC())
    app.bind('tcp://0.0.0.0:%d' % args.port)
    app.run()
