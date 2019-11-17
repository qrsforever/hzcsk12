#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-15 16:38:16

import os, sys, time
import argparse
import signal
import logging, json
import zerorpc
import requests
from threading import Thread

if os.environ.get("K12NLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

# TODO
LEVEL = logging.DEBUG

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

logger = logging.getLogger(__name__)

class AppServiceRPC(object):

    def rpctest(self):
        logger.error("call conntest ok!")

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

# def _loop(remote_api):
#     time.sleep(2)
#     while True:
#         try:
#             requests.get(remote_api)
#             time.sleep(100)
#         except Exception as e:
#             time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--host',
            default='0.0.0.0',
            type=str,
            dest='host',
            help="host to run app service")
    parser.add_argument(
            '--port',
            default=8149,
            type=int,
            dest='port',
            help="port to run app service")
    parser.add_argument(
            '--k12ai_host',
            default='0.0.0.0',
            type=str,
            dest='k12ai_host',
            help="k12ai_host to run app service")
    parser.add_argument(
            '--k12ai_port',
            default=8129,
            type=int,
            dest='k12ai_port',
            help="k12ai_port to run app service")
    args = parser.parse_args()

    logger.info('start zerorpc server on %s:%d', args.host, args.port)

    app = zerorpc.Server(AppServiceRPC())
    app.bind('tcp://%s:%d' % (args.host, args.port))

    # remote_api = 'http://%s:%d/k12ai/service/message'
    #         % (args.k12ai_host, args.k12ai_port, 'k12nlp', args.host, args.port)

    # Thread(target = _loop, args = (remote_api,)).start()

    app.run()
