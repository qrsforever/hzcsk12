#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12pyr_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-08-04 22:31

import os, time, re
import argparse
import json # noqa
import zerorpc
# from threading import Thread

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
            k12ai_consul_register('k12pyr', host, port)
            break
        except Exception as err:
            Logger.info("consul agent service register err: {}".format(err))
            time.sleep(3)


class PyrServiceRPC(ServiceRPC):

    def __init__(self, host, port, image, dataroot):
        super().__init__('pyr', host, port, image, dataroot, _DEBUG_)

        self._datadir = f'{dataroot}/datasets'
        self._pretrained_dir = f'{dataroot}/pretrained/cv'

    def errtype2errcode(self, op, user, uuid, errtype, errtext):
        errcode = 999999
        return errcode

    @k12ai_timeit(handler=Logger.info)
    def pre_processing(self, appId, op, user, uuid, params):
        usercache, innercache = self.get_cache_dir(user, uuid)
        self.oss_download(os.path.join(usercache, 'checkpoints'))
        if 'code' in params:
            for line in params['code'].split('\n'):
                line = line.strip()
                if not len(line) or line.startswith('#'):
                    continue
                # parse parameters to check whether constains oss
                if 'oss://' in line:
                    result = re.findall(r'oss://([a-zA-Z0-9\/\_\-]+)[\'\"]', line, re.S)
                    if len(result) > 0:
                        self.oss_download(os.path.join(usercache, result[0]))
        return params

    @k12ai_timeit(handler=Logger.info)
    def post_processing(self, appId, op, user, uuid, message):
        usercache, innercache = self.get_cache_dir(user, uuid)
        self.oss_upload(os.path.join(usercache, 'checkpoints'), clear=True)

    def make_container_volumes(self):
        volumes = {}
        volumes[self._pretrained_dir] = {'bind': '/root/.cache/torch/hub/checkpoints', 'mode': 'rw'}
        volumes[self._datadir] = {'bind': '/datasets', 'mode': 'rw'}
        if self._debug:
            litelib = 'pytorch-lightning/pytorch_lightning'
            volumes[f'{self._projdir}/app'] = {'bind': f'{self._workdir}/app', 'mode': 'rw'}
            volumes[f'{self._projdir}/{litelib}'] = {'bind': f'{self._workdir}/{litelib}', 'mode': 'rw'}
        return volumes

    def make_container_environs(self, op, params):
        environs = {}
        # environs['PYTHONPATH'] = '/hzcsk12/pyr/app'
        return environs

    def make_container_kwargs(self, op, params):
        kwargs = {}
        kwargs['runtime'] = 'nvidia'
        kwargs['shm_size'] = '10g'
        kwargs['mem_limit'] = '10g'
        return kwargs

    def make_container_command(self, appId, op, user, uuid, params):
        usercache, innercache = self.get_cache_dir(user, uuid)
        command = f'python3 {self._workdir}/app/k12ai/pyexec.py '
        if op.startswith('runcode') and 'code' in params:
            with open(os.path.join(usercache, 'pyrcode.py'), 'w') as fw:
                fw.write(params['code'])
            command += f'--pyfile {os.path.join(innercache, "pyrcode.py")}'
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
            default=8179,
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
            default='hzcsai_com/k12pyr',
            type=str,
            dest='image',
            help="image to run container")
    parser.add_argument(
            '--data_root',
            default='/root/data',
            type=str,
            dest='data_root',
            help="data root: datasets, pretrained, users")
    args = parser.parse_args()

    if _DEBUG_:
        k12ai_set_loglevel('debug')
    k12ai_set_logfile('k12pyr.log')

    k12ai_consul_init(args.consul_addr, args.consul_port, False)

    # thread = Thread(target=_delay_do_consul, args=(args.host, args.port))
    # thread.start()

    Logger.info(f'start zerorpc server on {args.host}:{args.port}')

    try:
        app = zerorpc.Server(PyrServiceRPC(
            host=args.host, port=args.port,
            image=args.image,
            dataroot=args.data_root))
        app.bind('tcp://0.0.0.0:%d' % (args.port))
        app.run()
    finally:
        g_app_quit = True
        # thread.join()
