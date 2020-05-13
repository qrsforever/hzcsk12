#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12cv_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-27 17:08:18

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
            k12ai_consul_register('k12cv', host, port)
            break
        except Exception as err:
            Logger.info("consul agent service register err: {}".format(err))
            time.sleep(3)


class CVServiceRPC(ServiceRPC):

    def __init__(self, host, port, image, dataroot):
        super().__init__('cv', host, port, image, dataroot, _DEBUG_)

        self._pretrained_dir = '%s/pretrained/cv' % dataroot

    def errtype2errcode(self, op, user, uuid, errtype, errtext):
        errcode = 999999
        if errtype == 'FileNotFoundError':
            if 'model file is not found' in errtext:
                errcode = 100208
        if errtype == 'ConfigMissingException':
            errcode = 100233
        elif errtype == 'InvalidModel':
            errcode = 100302
        elif errtype == 'InvalidOptimizerMethod':
            errcode = 100303
        elif errtype == 'InvalidPadMode':
            errcode = 100304
        elif errtype == 'InvalidAnchorMethod':
            errcode = 100305
        elif errtype == 'ImageTypeError':
            errcode = 100306
        elif errtype == 'TensorSizeError':
            errcode = 100307
        return errcode

    @k12ai_timeit(handler=Logger.info)
    def pre_processing(self, op, user, uuid, params):
        pass
        # params['data.data_dir']
        # self.oss_download('/data/datasets/cv/custom_mnist.tar.gz',
        #         prefix_map=['data/datasets/cv/', f'/data/users/{user}/{uuid}/'])

        # Logger.info('11111')
        # result = os.system(f'tar zxf /data/users/{user}/{uuid}/custom_mnist.tar.gz -C /data/users/{user}/{uuid}/')
        # if result != 0:
        #     Logger.error('xxx')
        # Logger.info('22222')

    @k12ai_timeit(handler=Logger.info)
    def post_processing(self, op, user, uuid, message):
        if op.startswith('train') and isinstance(message, dict):
            environs = self.get_container_environs(user, uuid)
            if environs:
                message['environ'] = {}
                message['environ']['dataset_name'] = environs['K12AI_DATASET_NAME']
                message['environ']['model_name'] = environs['K12AI_MODEL_NAME']
                message['environ']['batch_size'] = environs['K12AI_BATCH_SIZE']
                message['environ']['input_size'] = environs['K12AI_INPUT_SIZE']

        # save objects to oss
        output_data_dir = os.path.join(self.get_cache_dir(user, uuid), 'output')
        train_data_dir = os.path.join(output_data_dir, 'ckpts')
        evaluate_data_dir = os.path.join(output_data_dir, 'result')
        if op.startswith('train') and os.path.exists(train_data_dir) :
            self.oss_upload(train_data_dir, clear=True)
        elif op.startswith('evaluate') and os.path.exists(evaluate_data_dir):
            self.oss_upload(evaluate_data_dir, clear=True)

    def make_container_volumes(self):
        volumes = {}
        volumes[self._pretrained_dir] = {'bind': '/pretrained', 'mode': 'rw'}
        if self._debug:
            volumes[f'{self._projdir}/app'] = {'bind': f'{self._workdir}/app', 'mode': 'rw'}
            volumes[f'{self._projdir}/torchcv/data'] = {'bind': f'{self._workdir}/torchcv/data', 'mode': 'rw'}
            volumes[f'{self._projdir}/torchcv/metric'] = {'bind': f'{self._workdir}/torchcv/metric', 'mode': 'rw'}
            volumes[f'{self._projdir}/torchcv/model'] = {'bind': f'{self._workdir}/torchcv/model', 'mode': 'rw'}
            volumes[f'{self._projdir}/torchcv/runner'] = {'bind': f'{self._workdir}/torchcv/runner', 'mode': 'rw'}
            volumes[f'{self._projdir}/torchcv/lib/data'] = {'bind': f'{self._workdir}/torchcv/lib/data', 'mode': 'rw'}
            volumes[f'{self._projdir}/torchcv/lib/model'] = {'bind': f'{self._workdir}/torchcv/lib/model', 'mode': 'rw'}
            volumes[f'{self._projdir}/torchcv/lib/runner'] = {'bind': f'{self._workdir}/torchcv/lib/runner', 'mode': 'rw'}
            volumes[f'{self._projdir}/torchcv/lib/tools'] = {'bind': f'{self._workdir}/torchcv/lib/tools', 'mode': 'rw'}
            volumes[f'{self._projdir}/torchcv/main.py'] = {'bind': f'{self._workdir}/torchcv/main.py', 'mode': 'rw'}
        return volumes

    def make_container_environs(self, op, params):
        environs = {}
        if op.startswith('train'):
            environs['K12AI_DATASET_NAME'] = params['_k12.data.dataset_name']
            environs['K12AI_MODEL_NAME'] = params['network.backbone']
            environs['K12AI_BATCH_SIZE'] = params['train.batch_size']
            environs['K12AI_INPUT_SIZE'] = params['train.data_transformer.input_size']
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

    def make_container_command(self, op, user, uuid, params):
        cachedir = self.get_cache_dir(user, uuid)
        config_file = f'{cachedir}/config.json'
        if '_k12.data.dataset_name' in params.keys():
            config_tree = ConfigFactory.from_dict(params)
            _k12ai_tree = config_tree.pop('_k12')
            # Aug Trans
            if config_tree.get('train.aug_trans.trans_seq', default=None) is None:
                config_tree.put('train.aug_trans.trans_seq', [])
            if config_tree.get('val.aug_trans.trans_seq', default=None) is None:
                config_tree.put('val.aug_trans.trans_seq', [])
            if config_tree.get('test.aug_trans.trans_seq', default=None) is None:
                config_tree.put('test.aug_trans.trans_seq', [])
            for k, v in _k12ai_tree.get('trans_seq_group.train', default={}).items():
                if v == 'trans_seq':
                    config_tree.put('train.aug_trans.trans_seq', [k], append=True)
                if v == 'shuffle_trans_seq':
                    config_tree.put('train.aug_trans.shuffle_trans_seq', [k], append=True)
            for k, v in _k12ai_tree.get('trans_seq_group.val', default={}).items():
                if v == 'trans_seq':
                    config_tree.put('val.aug_trans.trans_seq', [k], append=True)
                if v == 'shuffle_trans_seq':
                    config_tree.put('val.aug_trans.shuffle_trans_seq', [k], append=True)
            for k, v in _k12ai_tree.get('trans_seq_group.test', default={}).items():
                if v == 'trans_seq':
                    config_tree.put('test.aug_trans.trans_seq', [k], append=True)
                if v == 'shuffle_trans_seq':
                    config_tree.put('test.aug_trans.shuffle_trans_seq', [k], append=True)
            # CheckPoints
            model_name = config_tree.get('network.model_name', default='unknow')
            backbone = config_tree.get('network.backbone', default='unknow')
            ckpts_name = '%s_%s_%s' % (model_name, backbone, _k12ai_tree.get('data.dataset_name'))
            config_tree.put('network.checkpoints_root', '/cache/output')
            config_tree.put('network.checkpoints_name', ckpts_name)
            config_tree.put('network.checkpoints_dir', 'ckpts')
            ckpts_file_exist = os.path.exists(f'{cachedir}/output/ckpts/{ckpts_name}_latest.pth')
            if op.startswith('evaluate'):
                if not ckpts_file_exist:
                    raise FileNotFoundError('model file is not found!')
                config_tree.put('network.resume_continue', True)
            else: # train
                if not ckpts_file_exist:
                    config_tree.put('network.resume_continue', False)
            if config_tree.get('network.resume_continue'):
                config_tree.put('network.resume', f'/cache/output/ckpts/{ckpts_name}_latest.pth')
            config_str = HOCONConverter.convert(config_tree, 'json')
        else:
            config_str = json.dumps(params)

        with open(config_file, 'w') as fout:
            fout.write(config_str)

        command = 'python %s/torchcv/main.py --config_file /cache/config.json' % self._workdir

        if op.startswith('train'):
            command += ' --phase train'
        elif op.startswith('evaluate'):
            command += ' --phase test --test_dir todo --out_dir /cache/output/result'
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
            default=8139,
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
            default='hzcsai_com/k12cv',
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
    k12ai_set_logfile('k12cv.log')

    k12ai_consul_init(args.consul_addr, args.consul_port, _DEBUG_)

    thread = Thread(target=_delay_do_consul, args=(args.host, args.port))
    thread.start()

    Logger.info(f'start zerorpc server on {args.host}:{args.port}')

    try:
        app = zerorpc.Server(CVServiceRPC(
            host=args.host, port=args.port,
            image=args.image,
            dataroot=args.data_root))
        app.bind('tcp://%s:%d' % (args.host, args.port))
        app.run()
    finally:
        g_app_quit = True
        thread.join()
