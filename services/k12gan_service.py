#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12gan_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-01-19

import os, time
import argparse
import json
import zerorpc
import re

from threading import Thread
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter

from k12ai.k12ai_base import ServiceRPC
from k12ai.k12ai_consul import (k12ai_consul_init, k12ai_consul_register)
from k12ai.k12ai_logger import (k12ai_set_loglevel, k12ai_set_logfile, Logger)
from k12ai.k12ai_utils import k12ai_timeit, mkdir_p
from k12ai.k12ai_platform import k12ai_platform_stats as get_platform_stats
from k12ai.k12ai_errmsg import FrameworkError

_DEBUG_ = True if os.environ.get("K12AI_DEBUG") else False

g_app_quit = False


class GanServiceRPC(ServiceRPC):

    def __init__(self, host, port, image, dataroot):
        super().__init__('gan', host, port, image, dataroot, _DEBUG_)

        self._datadir = f'{dataroot}/datasets/gan'
        self._pretrained_dir = f'{dataroot}/pretrained/gan'

    def errtype2errcode(self, op, user, uuid, errtype, errtext):
        errcode = 999999
        return errcode

    @k12ai_timeit(handler=Logger.info)
    def pre_processing(self, appId, op, user, uuid, params):
        return params

    @k12ai_timeit(handler=Logger.info)
    def post_processing(self, appId, op, user, uuid, message):
        return message

    def make_container_volumes(self):
        volumes = {}
        volumes[self._datadir] = {'bind': '/datasets', 'mode': 'r'}
        volumes[self._pretrained_dir] = {'bind': '/pretrained', 'mode': 'r'}
        if self._debug:
            volumes[f'{self._projdir}/app'] = {'bind': f'{self._workdir}/app', 'mode': 'rw'}
            volumes[f'{self._projdir}/pytorch-CycleGAN-and-pix2pix'] = {'bind': f'{self._workdir}/pytorch-CycleGAN-and-pix2pix', 'mode': 'rw'}
        return volumes

    def make_container_environs(self, op, params):
        environs = {}
        return environs

    def make_container_kwargs(self, op, params):
        kwargs = {}
        kwargs['runtime'] = 'nvidia'
        kwargs['shm_size'] = '10g'
        kwargs['mem_limit'] = '20g'
        return kwargs

    def make_container_command(self, appId, op, user, uuid, params):
        return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--host',
            default='127.0.0.1',
            type=str,
            dest='host',
            help="host to run app service")
    parser.add_argument(
            '--port',
            default=8239,
            type=int,
            dest='port',
            help="port to run app service")
    parser.add_argument(
            '--consul_addr',
            default='127.0.0.1',
            type=str,
            dest='consul_addr',
            help="consul address")
    parser.add_argument(
            '--consul_port',
            default=8500,
            type=int,
            dest='consul_port',
            help="consul port")
    parser.add_argument(
            '--image',
            default='hzcsai_com/k12gan',
            type=str,
            dest='image',
            help="image to run container")
    parser.add_argument(
            '--data_root',
            default='/data',
            type=str,
            dest='data_root',
            help="data root: datasets, pretrained, users")
    args = parser.parse_args()

    if _DEBUG_:
        k12ai_set_loglevel('debug')
    k12ai_set_logfile('k12gan.log')

    k12ai_consul_init(args.consul_addr, args.consul_port,  _DEBUG_)

    Logger.info(f'start zerorpc server on {args.host}:{args.port}')

    try:
        app = zerorpc.Server(GanServiceRPC(
            host=args.host, port=args.port,
            image=args.image,
            dataroot=args.data_root))
        app.bind('tcp://0.0.0.0:%d' % (args.port))
        app.run()
    finally:
        g_app_quit = True
        thread.join()
