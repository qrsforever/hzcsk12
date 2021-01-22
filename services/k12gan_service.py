#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12gan_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-01-19

import os
import argparse
import json
import zerorpc

from k12ai.k12ai_base import ServiceRPC
from k12ai.k12ai_consul import k12ai_consul_init
from k12ai.k12ai_logger import (k12ai_set_loglevel, k12ai_set_logfile, Logger)
from k12ai.k12ai_utils import k12ai_timeit
from k12ai.k12ai_errmsg import FrameworkError

_DEBUG_ = True if os.environ.get("K12AI_DEBUG") else False


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
        usercache, innercache = self.get_cache_dir(user, uuid)
        if 'train.start' == op:
            params['continue_train'] = False
        else:
            config = os.path.join(usercache, 'config.json')
            self.oss_download(config)
            if os.path.exists(config):
                with open(config, 'r') as fr:
                    _jdata = json.load(fr)
                    if params is not None:
                        _jdata.update(params)
                    params = _jdata
                params['continue_train'] = True
            else:
                raise FrameworkError(100214)
        if params['continue_train']:
            self.oss_download(os.path.join(usercache, 'ckpts'))

        return params

    @k12ai_timeit(handler=Logger.info)
    def post_processing(self, appId, op, user, uuid, message):
        usercache, innercache = self.get_cache_dir(user, uuid)
        if op.startswith('train'):
            self.oss_upload(os.path.join(usercache, 'config.json'), clear=True)
            self.oss_upload(os.path.join(usercache, 'ckpts'), clear=True)
        else: # op.startswith('evaluate')
            self.oss_upload(os.path.join(usercache, 'results'), clear=True)
        return message

    def make_container_volumes(self):
        volumes = {}
        volumes[self._datadir] = {'bind': '/datasets', 'mode': 'ro'}
        volumes[self._pretrained_dir] = {'bind': '/pretrained', 'mode': 'ro'}
        if self._debug:
            volumes[f'{self._projdir}/app'] = {'bind': f'{self._workdir}/app', 'mode': 'ro'}
            volumes[f'{self._projdir}/pytorch-CycleGAN-and-pix2pix'] = {'bind': f'{self._workdir}/pytorch-CycleGAN-and-pix2pix', 'mode': 'ro'}
        return volumes

    def make_container_kwargs(self, op, params):
        kwargs = {}
        kwargs['runtime'] = 'nvidia'
        kwargs['shm_size'] = '10g'
        kwargs['mem_limit'] = '10g'
        return kwargs

    def make_container_command(self, appId, op, user, uuid, params):
        # TODO
        params['continue_train'] = True
        Logger.info(params)
        usercache, innercache = self.get_cache_dir(user, uuid)
        config_file = f'{usercache}/config.json'
        with open(config_file, 'w') as fw:
            fw.write(json.dumps(params))

        if op.startswith('train'):
            command = f'python {self._workdir}/app/k12ai/train.py --phase train --config_file {innercache}/config.json'
        elif op.startswith('evaluate'):
            command = f'python {self._workdir}/app/k12ai/test.py --phase test --config_file {innercache}/config.json'
        elif op.startswith('predict'):
            command = f'python {self._workdir}/app/k12ai/predict.py --phase test --config_file {innercache}/config.json'
        else:
            raise NotImplementedError
        return command

    def clear_cache(self, user, uuid):
        pass


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

    k12ai_consul_init(args.consul_addr, args.consul_port, _DEBUG_)

    Logger.info(f'start zerorpc server on {args.host}:{args.port}')

    app = zerorpc.Server(GanServiceRPC(
        host=args.host, port=args.port,
        image=args.image,
        dataroot=args.data_root))
    app.bind('tcp://0.0.0.0:%d' % (args.port))
    app.run()
