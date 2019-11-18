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
import socket
import zerorpc
import requests
import consul
import tempfile
import subprocess
from threading import Thread
from allennlp.training.util import create_serialization_dir
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params

if os.environ.get("K12NLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

# TODO
LEVEL = logging.DEBUG

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

logger = logging.getLogger(__name__)

app_quit = False

def _get_host_name():
    return socket.gethostname()

def _get_host_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

app_host_name = _get_host_name()
app_host_ip = _get_host_ip()

consul_addr = None
consul_port = None

def _delay_do_consul(host, port):
    time.sleep(3)
    while not app_quit:
        try:
            client = consul.Consul(host=consul_addr, port=consul_port)
            client.agent.service.register(
                    name='{}-k12nlp'.format(socket.gethostname()),
                    address=host, port=port, tags=('AI', 'ML'),
                    check=consul.Check.tcp(host, port, interval='10s', timeout='20s', deregister='30s'))
            break
        except Exception as err:
            logger.info("consul agent service register err", err)
            time.sleep(3)

userdata_dir = '/data/users'

class NLPServiceRPC(object):

    def train(self, op, user, uuid, params):
        logger.info("call train")
        if op == 'stop':
            print(params['task_pid'])
            task_pid = str(params['task_pid'])
            if os.path.exists(os.path.join('/proc', task_pid)):
                subprocess.run(["kill", "-2", task_pid])
                return 1
            return 0
        pro_dir = os.path.join(userdata_dir, user, uuid)
        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)
        config_path = os.path.join(pro_dir, 'config.json')
        with open(config_path, 'w') as fout:
            fout.write(json.dumps(params))
        output_dir = os.path.join(pro_dir, 'output')
        if os.path.exists(output_dir):
            try:
                create_serialization_dir(Params.from_file(config_path), output_dir, True, False)
                flag = '--recover'
            except ConfigurationError as err:
                flag = '--force'
        else:
            flag = ' '
        task = subprocess.Popen([
            "allennlp", "train", config_path, flag,
            "--serialization-dir", output_dir,
            "--include-package", "hzcsnlp",
        ])
        return task.pid

    def evaluate(self, *args, **kwargs):
        logger.info("call evaluate")

    def predict(self, *args, **kwargs):
        logger.info("call predict")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--host',
            default=None,
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
            '--consul_addr',
            default=None,
            type=str,
            dest='consul_addr',
            help="consul address")
    parser.add_argument(
            '--consul_port',
            default=8500,
            type=int,
            dest='consul_port',
            help="consul port")
    args = parser.parse_args()

    logger.info('start zerorpc server on %s:%d', args.host, args.port)

    host = args.host if args.host else app_host_ip

    consul_addr = args.consul_addr if args.consul_addr else app_host_ip
    consul_port = args.consul_port

    thread = Thread(target=_delay_do_consul, args=(host, args.port))
    thread.start()

    try:
        app = zerorpc.Server(NLPServiceRPC())
        app.bind('tcp://%s:%d' % (host, args.port))
        app.run()
    finally:
        app_quit = True
        thread.join()
