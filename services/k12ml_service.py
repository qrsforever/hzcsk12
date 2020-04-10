#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ml_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-09 18:03

import os, time
import argparse
import json
import zerorpc
from threading import Thread
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter

from k12ai.k12ai_base import ServiceRPC
from k12ai.k12ai_consul import (k12ai_consul_init, k12ai_consul_register)
from k12ai.k12ai_logger import (k12ai_set_loglevel, k12ai_set_logfile, Logger)


_DEBUG_ = True if os.environ.get("K12AI_DEBUG") else False

g_app_quit = False


def _delay_do_consul(host, port):
    time.sleep(3)
    while not g_app_quit:
        try:
            k12ai_consul_register('k12ml', host, port)
            break
        except Exception as err:
            Logger.info("consul agent service register err: {}".format(err))
            time.sleep(3)


class MLServiceRPC(ServiceRPC):

    def __init__(self, host, port, image, dataroot):
        super().__init__('ml', host, port, image, dataroot, _DEBUG_)

    def errtype2errcode(self, errtype, errtext):
        if errtype == 'ConfigMissingException':
            errcode = 100233
        else:
            errcode = 999999
        return errcode

    def make_container_volumes(self):
        volumes = {}
        if self._debug:
            volumes[f'{self._projdir}/app'] = {'bind': f'{self._workdir}/app', 'mode':'rw'}
        return volumes

    def make_container_kwargs(self, op, params):
        kwargs = {
            'auto_remove': not self._debug,
            'runtime': 'nvidia',
            'shm_size': '2g',
            'mem_limit': '4g'
        }
        return kwargs

    def make_container_command(self, op, cachedir, params):
        config_file = f'{cachedir}/config.json'

        if '_k12.data.dataset_name' in params.keys():
            config_tree = ConfigFactory.from_dict(params)
            _k12ai_tree = config_tree.pop('_k12')
            for k, v in _k12ai_tree.get('metrics', default={}).items():
                if v and not config_tree.get('metrics.%s' % k, default=None):
                    config_tree.put('metrics.%s' % k, {})
            config_str = HOCONConverter.convert(config_tree, 'json')
        else:
            config_str = json.dumps(params)

        with open(config_file, 'w') as fout:
            fout.write(config_str)

        command = 'python {}'.format('%s/app/k12ai/main.py' % self._workdir)
        if op.startswith('train'):
            command += ' --phase train --config_file /cache/config.json'
        elif op.startswith('evaluate'):
            command += ' --phase evaluate --config_file /cache/config.json'
        else:
            raise NotImplementedError
        return 100000, command


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
            default=8129,
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
    parser.add_argument(
            '--image',
            default='hzcsai_com/k12ml',
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
    k12ai_set_logfile('k12ml.log')

    k12ai_consul_init(args.consul_addr, args.consul_port, _DEBUG_)

    thread = Thread(target=_delay_do_consul, args=(args.host, args.port))
    thread.start()

    Logger.info(f'start zerorpc server on {args.host}:{args.port}')

    try:
        app = zerorpc.Server(MLServiceRPC(
            host=args.host, port=args.port,
            image=args.image,
            dataroot=args.data_root
        ))
        app.bind('tcp://%s:%d' % (args.host, args.port))
        app.run()
    finally:
        g_app_quit = True
        thread.join()
