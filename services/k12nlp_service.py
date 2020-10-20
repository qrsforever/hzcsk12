#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-15 16:38:16

import os, time
import argparse
import zerorpc
from threading import Thread
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter

from k12ai.k12ai_base import ServiceRPC
from k12ai.k12ai_consul import (k12ai_consul_init, k12ai_consul_register)
from k12ai.k12ai_utils import k12ai_utils_diff
from k12ai.k12ai_logger import (k12ai_set_loglevel, k12ai_set_logfile, Logger)
from k12ai.k12ai_utils import k12ai_timeit

_DEBUG_ = True if os.environ.get("K12AI_DEBUG") else False

g_app_quit = False


def _delay_do_consul(host, port):
    time.sleep(3)
    while not g_app_quit:
        try:
            k12ai_consul_register('k12nlp', host, port)
            break
        except Exception as err:
            Logger.info("consul agent service register err: {}".format(err))
            time.sleep(3)


class NLPServiceRPC(ServiceRPC):

    def __init__(self, host, port, image, dataroot):
        super().__init__('nlp', host, port, image, dataroot, _DEBUG_)

        self._datadir = f'{dataroot}/datasets/nlp'
        self._pretrained_dir = '%s/pretrained/nlp' % dataroot
        self._nltk_data_dir = '%s/nltk_data' % dataroot

    def errtype2errcode(self, op, user, uuid, errtype, errtext):
        if errtype == 'FileNotFoundError':
            if 'model file is not found' in errtext:
                return 100208
            elif 'test file is not found' in errtext:
                return 100232
        return 999999

    @k12ai_timeit(handler=Logger.info)
    def pre_processing(self, appId, op, user, uuid, params):
        usercache, innercache = self.get_cache_dir(user, uuid)
        # download train data (weights)
        if params['_k12.model.resume'] or not op.startswith('train'):
            self.oss_download(os.path.join(usercache, 'output', 'traindata'))

        return params

    @k12ai_timeit(handler=Logger.info)
    def post_processing(self, appId, op, user, uuid, message):
        usercache, innercache = self.get_cache_dir(user, uuid)
        # upload train or evaluate data
        if op.startswith('train'):
            self.oss_upload(os.path.join(usercache, 'output', 'traindata'), clear=True)
        elif op.startswith('evaluate'):
            self.oss_upload(os.path.join(usercache, 'output', 'result'), clear=True)

    def make_container_volumes(self):
        volumes = {
            self._datadir: {'bind': '/datasets', 'mode': 'rw'},
            self._pretrained_dir: {'bind': '/pretrained', 'mode': 'rw'},
            self._nltk_data_dir: {'bind': '/root/nltk_data', 'mode': 'ro'}
        }
        if self._debug:
            volumes[f'{self._projdir}/app'] = {'bind': f'{self._workdir}/app', 'mode':'rw'}
            volumes[f'{self._projdir}/allennlp/allennlp'] = {'bind': f'{self._workdir}/allennlp/allennlp', 'mode':'rw'}
            volumes[f'{self._projdir}/allennlp-models/allennlp_models'] = {'bind': f'{self._workdir}/allennlp-models/allennlp_models', 'mode':'rw'}
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

        config_tree = ConfigFactory.from_dict(params)
        _k12ai_tree = config_tree.pop('_k12')
        resume = _k12ai_tree.get('model.resume', True)
        test_file = config_tree.get('test_data_path', '')
        config_str = HOCONConverter.convert(config_tree, 'json')

        with open(config_file, 'w') as fout:
            fout.write(config_str)

        if op.startswith('train'):
            flag = '--force'
            if resume:
                config_conf = os.path.join(usercache, 'config.json')
                serial_conf = os.path.join(usercache, 'output/traindata', 'config.json')
                if os.path.exists(serial_conf):
                    if not k12ai_utils_diff(config_conf, serial_conf):
                        flag = '--recover'
            command = f'allennlp train {innercache}/config.json %s --serialization-dir {innercache}/output/traindata' % flag
        elif op.startswith('evaluate'):
            model_file_outer = os.path.join(usercache, 'output/traindata', 'best.th')
            if not os.path.exists(model_file_outer):
                raise FileNotFoundError('model file is not found!')
            if not test_file:
                raise FileNotFoundError('test file is not found!')
            command = f'allennlp evaluate {innercache}/output/traindata {test_file}'
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
            default=8149,
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
            default='hzcsai_com/k12nlp',
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
    k12ai_set_logfile('k12nlp.log')

    k12ai_consul_init(args.consul_addr, args.consul_port, False)

    thread = Thread(target=_delay_do_consul, args=(args.host, args.port))
    thread.start()

    Logger.info(f'start zerorpc server on {args.host}:{args.port}')

    try:
        app = zerorpc.Server(NLPServiceRPC(
            host=args.host, port=args.port,
            image=args.image,
            dataroot=args.data_root
        ))
        app.bind('tcp://%s:%d' % (args.host, args.port))
        app.run()
    finally:
        g_app_quit = True
        thread.join()
