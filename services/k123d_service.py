#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k123d_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-06-22 10:25

import os, time
import argparse
import json # noqa
import zerorpc
from threading import Thread
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter

from k12ai.k12ai_base import ServiceRPC
from k12ai.k12ai_consul import (k12ai_consul_init, k12ai_consul_register)
from k12ai.k12ai_logger import (k12ai_set_loglevel, k12ai_set_logfile, Logger)
from k12ai.k12ai_utils import k12ai_timeit

_DEBUG_ = True if os.environ.get("K12AI_DEBUG") else False

g_app_quit = False


def _delay_do_consul(host, port):
    time.sleep(3)
    while not g_app_quit:
        try:
            k12ai_consul_register('k123d', host, port)
            break
        except Exception as err:
            Logger.info("consul agent service register err: {}".format(err))
            time.sleep(3)


class CV3DServiceRPC(ServiceRPC):

    def __init__(self, host, port, image, dataroot):
        super().__init__('3d', host, port, image, dataroot, _DEBUG_)

        self._datadir = f'{dataroot}/datasets/3d'
        self._pretrained_dir = f'{dataroot}/pretrained/cv'

    def errtype2errcode(self, op, user, uuid, errtype, errtext):
        errcode = 999999
        return errcode

    @k12ai_timeit(handler=Logger.info)
    def pre_processing(self, appId, op, user, uuid, params):
        return params

    @k12ai_timeit(handler=Logger.info)
    def post_processing(self, op, user, uuid, message):
        pass

    def make_container_volumes(self):
        volumes = {}
        volumes[self._datadir] = {'bind': '/datasets', 'mode': 'rw'},
        volumes[self._pretrained_dir] = {'bind': '/root/.cache/torch/hub/checkpoints', 'mode': 'rw'}
        if self._debug:
            volumes[f'{self._projdir}/app'] = {'bind': f'{self._workdir}/app', 'mode': 'rw'}
        return volumes

    def make_container_environs(self, op, params):
        environs = {}
        environs['PYTHONPATH'] = '/hzcsk12/3d/app'
        return environs

    def make_container_kwargs(self, op, params):
        kwargs = {}
        kwargs['runtime'] = 'nvidia'
        kwargs['shm_size'] = '10g'
        kwargs['mem_limit'] = '10g'
        return kwargs

    def get_app_memstat(self, params):
        # TODO
        bs = params['train.batch_size']
        dn = params['_k12.data.dataset_name']
        mm = 1
        if dn == 'dogsVsCats':
            mm = 2
        elif dn == 'aliproducts':
            mm = 2.5
        if bs <= 32:
            gmem = 4500 * mm
        elif bs == 64:
            gmem = 5500 * mm
        elif bs == 128:
            gmem = 6000 * mm
        else:
            gmem = 10000 * mm
        return {
            'app_cpu_memory_usage_MB': 6000,
            'app_gpu_memory_usage_MB': gmem,
        }

    def make_container_command(self, appId, op, user, uuid, params):
        usercache, innercache = self.get_cache_dir(user, uuid)
        config_file = f'{usercache}/config.json'

        with open(config_file, 'w') as fout:
            config_tree = ConfigFactory.from_dict(params)
            fout.write(HOCONConverter.convert(config_tree, 'json'))

        command = f'python {self._workdir}/app/k12ai/main.py --config_file {innercache}/config.json'

        if op.startswith('train'):
            command += ' --phase train'
        elif op.startswith('evaluate'):
            raise NotImplementedError
        elif op.startswith('predict'):
            raise NotImplementedError
        else:
            raise NotImplementedError
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
            default=8169,
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
            default='hzcsai_com/k123d',
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
    k12ai_set_logfile('k123d.log')

    k12ai_consul_init(args.consul_addr, args.consul_port, _DEBUG_)

    thread = Thread(target=_delay_do_consul, args=(args.host, args.port))
    thread.start()

    Logger.info(f'start zerorpc server on {args.host}:{args.port}')

    try:
        app = zerorpc.Server(CV3DServiceRPC(
            host=args.host, port=args.port,
            image=args.image,
            dataroot=args.data_root))
        app.bind('tcp://%s:%d' % (args.host, args.port))
        app.run()
    finally:
        g_app_quit = True
        thread.join()
