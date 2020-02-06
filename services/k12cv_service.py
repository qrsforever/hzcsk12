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
import _jsonnet
import zerorpc
import docker
from threading import Thread
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter

from k12ai_consul import k12ai_consul_init, k12ai_consul_register, k12ai_consul_message
from k12ai_utils import k12ai_utils_topdir
from k12ai_errmsg import k12ai_error_message as _err_msg
from k12ai_misc import PRETRAINED_MODELS
from k12ai_logger import (k12ai_set_loglevel, k12ai_set_logfile, Logger)

_DEBUG_ = True if os.environ.get("K12CV_DEBUG") else False

service_name = 'k12cv'

g_app_quit = False


def _delay_do_consul(host, port):
    time.sleep(3)
    while not g_app_quit:
        try:
            k12ai_consul_register(service_name, host, port)
            break
        except Exception as err:
            Logger.info("consul agent service register err: {}".format(err))
            time.sleep(3)


class CVServiceRPC(object):

    def __init__(self,
            host, port,
            image='hzcsai_com/k12cv',
            data_root='/data',
            workdir='/hzcsk12/cv'):
        self._debug = _DEBUG_
        self._host = host
        self._port = port
        self._image = image
        self._docker = docker.from_env()
        self._workdir = workdir
        self._projdir = os.path.join(k12ai_utils_topdir(), 'cv')
        Logger.info('workdir:%s, projdir:%s' % (self._workdir, self._projdir))

        self.userscache_dir = '%s/users' % data_root
        self.datasets_dir = '%s/datasets/cv' % data_root
        self.pretrained_dir = '%s/pretrained/cv' % data_root

    def send_message(self, op, user, uuid, msgtype, message, clear=False):
        if not msgtype:
            return
        if isinstance(message, dict):
            if 'err_type' in message:
                errtype = message['err_type']
                if errtype == 'ModelFileNotFound':
                    code = 100208
                elif errtype == 'InvalidModel':
                    code = 100302
                elif errtype == 'InvalidOptimizerMethod':
                    code = 100303
                elif errtype == 'InvalidPadMode':
                    code = 100304
                elif errtype == 'InvalidAnchorMethod':
                    code = 100305
                elif errtype == 'ImageTypeError':
                    code = 100306
                elif errtype == 'TensorSizeError':
                    code = 100307
                elif errtype == 'MemoryError':
                    code = 100901
                elif errtype == 'NotImplementedError':
                    code = 100902
                elif errtype == 'ConfigurationError':
                    code = 100903
                else:
                    code = 100399
                message = _err_msg(code, exc_info=message)
        k12ai_consul_message(user, op, 'k12cv', uuid, msgtype, message, clear)

    def _get_container(self, user, uuid):
        try:
            cons = self._docker.containers.list(all=True, filters={'label': [
                'k12ai.service.user=%s'%user,
                'k12ai.service.uuid=%s'%uuid]})
            if len(cons) == 1:
                return cons[0]
        except docker.errors.NotFound:
            pass
        return None

    def _get_cache_dir(self, user, uuid):
        usercache = '%s/%s/%s' % (self.userscache_dir, user, uuid)
        if not os.path.exists(usercache):
            os.makedirs(usercache)
        return usercache

    def _prepare_environ(self, phase, user, uuid, params):
        if not isinstance(params, dict):
            return 100231, 'parameters type is not dict'

        config_file = '%s/config.json' % self._get_cache_dir(user, uuid)

        # if phase == 'evaluate' and len(params) == 0:
        #     if os.path.exists(config_file):
        #         return 100000, None
        #     return 100209, f'config file:{config_file} not found'

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
            config_tree.put('network.checkpoints_root', '/cache')
            config_tree.put('network.checkpoints_name', ckpts_name)
            config_tree.put('network.checkpoints_dir', 'ckpts')

            if phase != 'train':
                config_tree.put('network.resume_continue', True)
            if config_tree.get('network.resume_continue', default=False):
                resume_path = '%s/ckpts/%s_latest.pth' % (self._get_cache_dir(user, uuid), ckpts_name)
                if os.path.exists(resume_path):
                    config_tree.put('network.resume', '/cache/ckpts/%s_latest.pth' % ckpts_name)

            # Pretrained
            pretrained = config_tree.get('network.pretrained', default=False)
            config_tree.pop('network.pretrained', default=None)
            if pretrained:
                _file = PRETRAINED_MODELS.get(backbone, 'nofile')
                if os.path.exists('%s/%s' % (self.pretrained_dir, _file)):
                    config_tree.put('network.pretrained', '/pretrained/%s' % _file)

            config_str = HOCONConverter.convert(config_tree, 'json')
        else:
            config_str = json.dumps(params)

        with open(config_file, 'w') as fout:
            fout.write(config_str)

        return 100000, None

    def _run(self, op, user, uuid, command=None):
        Logger.info(command)
        message = None
        rm_flag = True
        labels = { # noqa
                'k12ai.service.name': service_name,
                'k12ai.service.op': op,
                'k12ai.service.user': user,
                'k12ai.service.uuid': uuid
                }

        usercache_dir = self._get_cache_dir(user, uuid)

        volumes = { # noqa
                self.datasets_dir: {'bind': '/datasets', 'mode': 'rw'},
                usercache_dir: {'bind': '/cache', 'mode': 'rw'},
                self.pretrained_dir: {'bind': '/pretrained', 'mode': 'rw'},
                }

        if self._debug:
            rm_flag = False
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

        environs = {
                'K12CV_RPC_HOST': '%s' % self._host,
                'K12CV_RPC_PORT': '%s' % self._port,
                'K12CV_OP': '%s' % op,
                'K12CV_USER': '%s' % user,
                'K12CV_UUID': '%s' % uuid
                } # noqa
        kwargs = {
                'name': '%s-%s-%s' % (op, user, uuid),
                'auto_remove': rm_flag,
                'detach': True,
                'runtime': 'nvidia',
                'labels': labels,
                'volumes': volumes,
                'environment': environs,
                'shm_size': '4g',
                'mem_limit': '8g',
                } # noqa

        self.send_message(op, user, uuid, "status", {'value': 'starting'}, clear=True)
        try:
            self._docker.containers.run(self._image, command, **kwargs)
            return
        except Exception:
            message = _err_msg(100203, f'container image:{self._image}', exc=True)
            self.send_message(op, user, uuid, "status", {'value': 'exit', 'way': 'docker'})

        if message:
            self.send_message(op, user, uuid, "error", message)

    def schema(self, task, netw, dataset_name):
        schema_file = os.path.join(self._projdir, 'app', 'templates', 'schema', 'k12ai_cv.jsonnet')
        if not os.path.exists(schema_file):
            return 100206, f'{schema_file}'
        schema_json = _jsonnet.evaluate_file(schema_file, ext_vars={
            'task': task,
            'network': netw,
            'dataset_name': dataset_name})
        return 100000, json.dumps(json.loads(schema_json), separators=(',', ':'))

    def execute(self, op, user, uuid, params):
        Logger.info("call execute(%s, %s, %s)" % (op, user, uuid))
        container = self._get_container(user, uuid)
        phase, action = op.split('.')
        if action == 'stop':
            if container is None or container.status != 'running':
                return 100205, None
            container.kill()
            self.send_message('%s.start' % phase, user, uuid, "status", {'value': 'exit', 'way': 'manual'})
            return 100000, None

        if container:
            if container.status == 'running':
                return 100204, None
            container.remove()

        code, result = self._prepare_environ(phase, user, uuid, params)
        if code != 100000:
            return code, result

        command = 'python -m torch.distributed.launch --nproc_per_node=1 {}'.format(
                '%s/torchcv/main.py' % self._workdir)

        command += ' --config_file /cache/config.json'

        if phase == 'train':
            command += ' --phase train'
        elif phase == 'evaluate':
            command += ' --phase test --test_dir todo --out_dir /cache/output'
        elif phase == 'predict':
            raise('not impl yet')

        Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
            daemon=True).start()
        return 100000, None


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
            data_root=args.data_root
        ))
        app.bind('tcp://%s:%d' % (args.host, args.port))
        app.run()
    finally:
        g_app_quit = True
        thread.join()
