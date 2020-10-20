#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12rl_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-19 15:55

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
from k12ai.k12ai_utils import k12ai_timeit

_DEBUG_ = True if os.environ.get("K12AI_DEBUG") else False

g_app_quit = False


def _delay_do_consul(host, port):
    time.sleep(3)
    while not g_app_quit:
        try:
            k12ai_consul_register('k12rl', host, port)
            break
        except Exception as err:
            Logger.info("consul agent service register err: {}".format(err))
            time.sleep(3)


class RLServiceRPC(ServiceRPC):

    def __init__(self, host, port, image, dataroot):
        super().__init__('rl', host, port, image, dataroot, False)

        self._datadir = f'{dataroot}/datasets/rl'

    def errtype2errcode(self, op, user, uuid, errtype, errtext):
        errcode = 999999
        if errtype == 'ConfigMissingException':
            errcode = 100233
        elif errtype == 'FileNotFoundError':
            if errtext == 'k12ai: snapshot file is broken!':
                errcode = 100211
            elif errtext == 'k12ai: snapshot file is not found!':
                errcode = 100208
        return errcode

    @k12ai_timeit(handler=Logger.info)
    def pre_processing(self, appId, op, user, uuid, params):
        usercache, innercache = self.get_cache_dir(user, uuid)
        # download train data (weights)
        if params['_k12.model.resume'] or not op.startswith('train'):
            self.oss_download(os.path.join(usercache, 'output', 'run_snap'))
    
        return params

    @k12ai_timeit(handler=Logger.info)
    def post_processing(self, appId, op, user, uuid, message):
        usercache, innercache = self.get_cache_dir(user, uuid)
        # upload train or evaluate data
        if op.startswith('train'):
            self.oss_upload(os.path.join(usercache, 'output', 'run_snap'), clear=True)
        elif op.startswith('evaluate'):
            self.oss_upload(os.path.join(usercache, 'output', 'result'), clear=True)

    def make_container_volumes(self):
        volumes = {}
        volumes[self._datadir] = {'bind': '/datasets', 'mode': 'rw'},
        if self._debug:
            volumes[f'{self._projdir}/app'] = {'bind': f'{self._workdir}/app', 'mode':'rw'}
            volumes[f'{self._projdir}/rlpyt'] = {'bind': f'{self._workdir}/rlpyt', 'mode': 'rw'}
        return volumes

    def make_container_kwargs(self, op, params):
        kwargs = {
            'shm_size': '4g',
            'mem_limit': '8g'
        }
        return kwargs

    def make_container_command(self, appId, op, user, uuid, params):
        usercache, innercache = self.get_cache_dir(user, uuid)
        config_file = f'{usercache}/config.json'

        if '_k12.task' in params.keys():
            config_tree = ConfigFactory.from_dict(params)
            config_str = HOCONConverter.convert(config_tree, 'json')
        else:
            config_str = json.dumps(params)

        with open(config_file, 'w') as fout:
            fout.write(config_str)

        command = 'main.sh'
        if op.startswith('train'):
            command += f' --phase train --config_file {innercache}/config.json'
        elif op.startswith('evaluate'):
            command += f' --phase evaluate --config_file {innercache}/config.json'
        else:
            raise NotImplementedError
        return command


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
            default=8139,
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
            default='hzcsai_com/k12rl',
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
    k12ai_set_logfile('k12rl.log')

    k12ai_consul_init(args.consul_addr, args.consul_port, _DEBUG_)

    thread = Thread(target=_delay_do_consul, args=(args.host, args.port))
    thread.start()

    Logger.info(f'start zerorpc server on {args.host}:{args.port}')

    try:
        app = zerorpc.Server(RLServiceRPC(
            host=args.host, port=args.port,
            image=args.image,
            dataroot=args.data_root
        ))
        app.bind('tcp://%s:%d' % (args.host, args.port))
        app.run()
    finally:
        g_app_quit = True
        thread.join()
