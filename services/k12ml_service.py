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
import _jsonnet
import zerorpc
import docker
from threading import Thread
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter

from k12ai_consul import (k12ai_consul_init, k12ai_consul_register, k12ai_consul_message)
from k12ai_utils import (k12ai_utils_topdir, k12ai_utils_netip)
from k12ai_errmsg import k12ai_error_message as _err_msg
from k12ai_logger import (k12ai_set_loglevel, k12ai_set_logfile, Logger)
from k12ai_platform import (k12ai_platform_cpu_count, k12ai_platform_gpu_count)

_DEBUG_ = True if os.environ.get("K12AI_DEBUG") else False

service_name = 'k12ml'

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


class MLServiceRPC(object):

    def __init__(self,
            host, port,
            image='hzcsai_com/k12ml',
            data_root='/data',
            workdir='/hzcsk12/ml'):
        self._debug = _DEBUG_
        self._host = host
        self._port = port
        self._netip = k12ai_utils_netip()
        self._cpu_count = k12ai_platform_cpu_count()
        self._gpu_count = k12ai_platform_gpu_count()
        self._image = image
        self._docker = docker.from_env()
        self._workdir = workdir
        self._projdir = os.path.join(k12ai_utils_topdir(), 'ml')
        self._commlib = os.path.join(k12ai_utils_topdir(), 'common', '_delta_')
        Logger.info('workdir:%s, projdir:%s' % (self._workdir, self._projdir))

        self.userscache_dir = '%s/users' % data_root
        self.datasets_dir = '%s/datasets/ml' % data_root

    def send_message(self, token, op, user, uuid, msgtype, message, clear=False):
        if not msgtype:
            return
        if isinstance(message, dict):
            if 'err_type' in message:
                errtype = message['err_type']
                if errtype == 'ConfigMissingException':
                    code = 100233
                elif errtype == 'MemoryError':
                    code = 100901
                elif errtype == 'NotImplementedError':
                    code = 100902
                elif errtype == 'ConfigurationError':
                    code = 100903
                elif errtype == 'FileNotFoundError':
                    code = 100905
                else:
                    code = 100999
                message = _err_msg(code, exc_info=message)
        k12ai_consul_message(token, user, op, 'k12ml', uuid, msgtype, message, clear)

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

    def _prepare_environ(self, user, uuid, params):
        if not isinstance(params, dict):
            return 100231, 'parameters type is not dict'

        if '_k12.data.dataset_name' in params.keys():
            config_tree = ConfigFactory.from_dict(params)
            _k12ai_tree = config_tree.pop('_k12')
            for k, v in _k12ai_tree.get('metrics', default={}).items():
                if v and not config_tree.get('metrics.%s' % k, default=None):
                    config_tree.put('metrics.%s' % k, {})
            config_str = HOCONConverter.convert(config_tree, 'json')
        else:
            config_str = json.dumps(params)

        config_file = '%s/config.json' % self._get_cache_dir(user, uuid)
        with open(config_file, 'w') as fout:
            fout.write(config_str)

        return 100000, None

    def _run(self, token, op, user, uuid, command=None):
        Logger.info(command)
        message = None
        rm_flag = True
        labels = {
            'k12ai.service.name': service_name,
            'k12ai.service.op': op,
            'k12ai.service.user': user,
            'k12ai.service.uuid': uuid
        }

        usercache_dir = self._get_cache_dir(user, uuid)

        volumes = {
            self.datasets_dir: {'bind': '/datasets', 'mode': 'rw'},
            usercache_dir: {'bind': '/cache', 'mode': 'rw'},
        }

        if self._debug:
            rm_flag = False
            volumes[f'{self._projdir}/app'] = {'bind': f'{self._workdir}/app', 'mode': 'rw'}
            volumes[f'{self._projdir}/mla/mla'] = {'bind': f'{self._workdir}/mla', 'mode': 'rw'}

            volumes[f'{self._commlib}'] = {'bind': f'{self._workdir}/app/k12ai/common', 'mode': 'rw'}

        environs = {
            'K12AI_RPC_HOST': '%s' % self._host,
            'K12AI_RPC_PORT': '%s' % self._port,
            'K12AI_TOKEN': '%s' % token,
            'K12AI_OP': '%s' % op,
            'K12AI_USER': '%s' % user,
            'K12AI_UUID': '%s' % uuid
        }
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
        }

        self.send_message(token, op, user, uuid, "status", {'value': 'starting'}, clear=True)
        try:
            self._docker.containers.run(self._image, command, **kwargs)
            return
        except Exception:
            message = _err_msg(100302, 'container image:{}'.format(self._image), exc=True)
            self.send_message(token, op, user, uuid, "status", {'value': 'exit', 'way': 'docker'})

        if message:
            self.send_message(token, op, user, uuid, "error", message)

    def schema(self, task, netw, dataset_name):
        schema_file = os.path.join(self._projdir, 'app', 'templates', 'schema', 'k12ai_ml.jsonnet')
        if not os.path.exists(schema_file):
            return 100206, f'{schema_file}'
        schema_json = _jsonnet.evaluate_file(schema_file,
            ext_vars={
                'net_ip': self._netip,
                'task': task,
                'network': netw,
                'dataset_name': dataset_name},
            ext_codes={
                'debug': 'true' if self._debug else 'false',
                'num_cpu': str(self._cpu_count),
                'num_gpu': str(self._gpu_count)})
        return 100000, json.dumps(json.loads(schema_json), separators=(',', ':'))

    def execute(self, token, op, user, uuid, params):
        Logger.info("call execute(%s, %s, %s)" % (op, user, uuid))
        container = self._get_container(user, uuid)
        phase, action = op.split('.')
        if action == 'stop':
            if container is None or container.status != 'running':
                return 100205, None
            container.kill()
            self.send_message(token, '%s.start' % phase, user, uuid, "status", {'value': 'exit', 'way': 'manual'})
            return 100000, None

        if container:
            if container.status == 'running':
                return 100204, None
            container.remove()

        code, result = self._prepare_environ(user, uuid, params)
        if code != 100000:
            return code, result

        command = 'python {}'.format('%s/app/k12ml/main.py' % self._workdir)
        command += ' --phase %s --config_file /cache/config.json' % phase
        Thread(target=lambda: self._run(token=token, op=op, user=user, uuid=uuid, command=command),
            daemon=True).start()
        return 100000, None


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
            data_root=args.data_root
        ))
        app.bind('tcp://%s:%d' % (args.host, args.port))
        app.run()
    finally:
        g_app_quit = True
        thread.join()
