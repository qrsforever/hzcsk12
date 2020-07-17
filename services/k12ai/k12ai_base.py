#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-02 00:31

import os
import shutil
import json, _jsonnet
import docker
import time
from threading import Thread

from k12ai.k12ai_utils import (k12ai_utils_topdir, k12ai_utils_netip)
from k12ai.k12ai_errmsg import (k12ai_error_message, gen_exc_info)
from k12ai.k12ai_consul import k12ai_consul_message
from k12ai.k12ai_logger import Logger
from k12ai.k12ai_platform import (k12ai_platform_cpu_count, k12ai_platform_gpu_count, k12ai_platform_memory_free)
from k12ai.k12ai_utils import ( # noqa
        k12ai_oss_client, k12ai_object_remove,
        k12ai_object_get, k12ai_object_put)

MAX_TASKS = 10


class ServiceRPC(object):

    def __init__(self, sname, host, port, image, dataroot, debug):
        self._docker = docker.from_env()
        self._ossmc = k12ai_oss_client()
        self._sname = sname
        self._image = image
        self._host = host
        self._port = port
        self._debug = debug
        self._workdir = f'/hzcsk12/{self._sname}'
        self._projdir = os.path.join(k12ai_utils_topdir(), self._sname)

        self._netip = k12ai_utils_netip()
        self._cpu_count = k12ai_platform_cpu_count()
        self._gpu_count = k12ai_platform_gpu_count()

        self._datadir = f'{dataroot}/datasets/{self._sname}'
        self._userdir = f'{dataroot}/users'
        self._commlib = os.path.join(k12ai_utils_topdir(), 'services', 'k12ai', 'common')
        self._jschema = os.path.join(self._projdir, 'app', 'templates', 'schema')

    def send_message(self, token, op, user, uuid, msgtype, message, clear=False):
        if not msgtype:
            return
        if msgtype == 'error':
            errcode = 999999
            if 'errinfo' in message:
                errinfo = message['errinfo']
                if 'warning' == message['status']:
                    errcode = 100005
                elif isinstance(errinfo, dict) and 'err_type' in errinfo and 'err_text' in errinfo:
                    errcode = self.container_on_crash(op, user, uuid, message)
            else:
                # Status
                if 'starting' == message['status']:
                    errcode = 100001
                elif 'running' == message['status']:
                    errcode = 100002
                elif 'stop' == message['status']:
                    errcode = self.container_on_stop(op, user, uuid, message)
                elif 'finish' == message['status']:
                    errcode = self.container_on_finished(op, user, uuid, message)
            message = k12ai_error_message(errcode, expand=message)

        if isinstance(message, dict):
            if len(json.dumps(message)) > 500000: # max 512kb
                errcode = 100011
                message = k12ai_error_message(errcode)
        k12ai_consul_message(f'k12{self._sname}', token, op, user, uuid, msgtype, message, clear)

    def container_on_crash(self, op, user, uuid, message):
        errtype = message['errinfo']['err_type']
        errtext = message['errinfo']['err_text']
        errcode = self.errtype2errcode(op, user, uuid, errtype, errtext)
        if errcode == 999999:
            if errtype == 'MemoryError':
                errcode = 100901
            elif errtype == 'NotImplementedError':
                errcode = 100902
            elif errtype == 'ConfigurationError':
                errcode = 100903
            elif errtype == 'FileNotFoundError':
                errcode = 100905
            elif errtype == 'LossNanError':
                errcode = 100907
            elif errtype == 'RuntimeError':
                if 'CUDA out of memory' in errtext:
                    errcode = 100908
                elif 'systemcall error' in errtext:
                    errcode = 100909
            elif errtype == 'ImageNotFound':
                errcode = 100905
        self.on_finished(op, user, uuid, message)
        return errcode

    def container_on_stop(self, op, user, uuid, message):
        self.on_finished(op, user, uuid, message)
        return 100004

    def container_on_finished(self, op, user, uuid, message):
        self.on_finished(op, user, uuid, message)
        return 100003

    def pre_processing(self, op, user, uuid, params):
        pass

    def post_processing(self, op, user, uuid, message):
        pass

    def on_starting(self, op, user, uuid, params):
        self.clear_cache(user, uuid)
        self.pre_processing(op, user, uuid, params)

    def on_finished(self, op, user, uuid, message):
        self.post_processing(op, user, uuid, message)
        if not user.startswith('1660154'):
            self.clear_cache(user, uuid)

    def oss_upload(self, filepath, bucket_name=None, prefix_map=None, clear=False):
        if not os.path.exists(filepath):
            return
        try:
            if clear:
                k12ai_object_remove(self._ossmc, remote_path=filepath)
            result = k12ai_object_put(self._ossmc, local_path=filepath,
                    bucket_name=bucket_name, prefix_map=prefix_map)
            Logger.info(result)
        except Exception as err:
            Logger.error(str(err))

    def oss_download(self, filepath, bucket_name=None, prefix_map=None):
        try:
            result = k12ai_object_get(self._ossmc, remote_path=filepath,
                    bucket_name=bucket_name, prefix_map=prefix_map)
            Logger.info(result)
        except Exception as err:
            Logger.error(str(err))

    def make_container_command(self, op, cachedir, params):
        raise NotImplementedError

    def make_container_labels(self):
        return {}

    def make_container_volumes(self):
        return {}

    def make_container_environs(self, op, params):
        return {}

    def make_container_kwargs(self, op, params):
        return {}

    def make_schema_kwargs(self):
        return {}, {}

    def get_container(self, user, uuid):
        try:
            cons = self._docker.containers.list(all=True, filters={'label': [
                f'k12ai.service.user={user}',
                f'k12ai.service.uuid={uuid}']})
            if len(cons) == 1:
                return cons[0]
        except docker.errors.NotFound:
            pass
        return None

    def get_container_environs(self, user, uuid):
        con = self.get_container(user, uuid)
        if not con:
            return None
        environs = {}
        for item in con.attrs['Config']['Env']:
            if not item.startswith('K12AI_'):
                continue
            key, val = item.split('=')
            environs[key] = val
        return environs

    def clear_cache(self, user, uuid):
        usercache = os.path.join(self._userdir, user, uuid)
        try:
            shutil.rmtree(usercache)
            os.removedirs(os.path.dirname(usercache))
        except Exception:
            pass

    def get_cache_dir(self, user, uuid):
        usercache = os.path.join(self._userdir, user, uuid)
        if not os.path.exists(usercache):
            os.makedirs(usercache)
        return usercache, '/cache'

    def get_app_memstat(self, params):
        return {
            'app_cpu_memory_usage_MB': 6000,
            'app_gpu_memory_usage_MB': 6000,
        }

    def start_container_worker(self, token, op, user, uuid, params):
        usercache, innercache = self.get_cache_dir(user, uuid)
        try:
            self.on_starting(op, user, uuid, params)

            tb_logdir = None
            if '_k12.tb_logdir' in params.keys():
                tb_logdir = params['_k12.tb_logdir']

            labels = {
                'k12ai.service.name': f'k12{self._sname}',
                'k12ai.service.op': op,
                'k12ai.service.user': user,
                'k12ai.service.uuid': uuid,
                **self.make_container_labels()
            }

            volumes = {
                usercache: {'bind': innercache, 'mode': 'rw'},
                self._datadir: {'bind': '/datasets', 'mode': 'rw'},
                self._commlib: {'bind': f'{self._workdir}/app/k12ai/common', 'mode': 'rw'},
                **self.make_container_volumes()
            }

            if tb_logdir:
                volumes['/data/tblogs'] = {'bind': '/data/tblogs', 'mode':'rw'}

            environs = {
                'K12AI_RPC_HOST': self._host,
                'K12AI_RPC_PORT': self._port,
                'K12AI_TOKEN': token,
                'K12AI_OP': op,
                'K12AI_USER': user,
                'K12AI_UUID': uuid,
                **self.make_container_environs(op, params)
            }

            if tb_logdir:
                environs['K12AI_TBLOG_DIR'] = tb_logdir

            # W: don't set hostname
            kwargs = {
                'name': '%s-%s-%s' % (op, user, uuid),
                'auto_remove': False if tb_logdir else True,
                'detach': True,
                'runtime': 'nvidia',
                'labels': labels,
                'volumes': volumes,
                'environment': environs,
                **self.make_container_kwargs(op, params)
            }

            self.send_message(token, op, user, uuid, "error", {'status': 'starting'}, clear=True)
            command = self.make_container_command(op, user, uuid, params)
            self._docker.containers.run(f'{self._image}', command, **kwargs)
            return
        except Exception as err:
            Logger.error(str(err))
            self.send_message(token, op, user, uuid, "error", {
                'status': 'crash', 'errinfo': gen_exc_info()
            })

    def stop_container_worker(self, token, op, user, uuid, container):
        op = op.replace('stop', 'start')
        try:
            container.kill()
            self.send_message(token, op, user, uuid, "error", {'status': 'stop', 'event': 'by manual way'})
        except Exception as err:
            Logger.error(str(err))
            self.send_message(token, op, user, uuid, "error", {
                'status': 'crash', 'errinfo': gen_exc_info()
            })

    def schema(self, version, levelid, task, netw, dname, dinfo=None):
        Logger.info(f'{task}, {netw}, {dname}, {dinfo}')
        if not os.path.exists(self._jschema):
            return 100206, f'{self._jschema}'
        version_file = os.path.join(self._jschema, 'version.txt')
        with open(version_file, 'r') as fp:
            curver = str(fp.readlines()[0]).strip()
            if curver == version:
                return 100010, None
        jsonnet_file = os.path.join(self._jschema, 'k12ai.jsonnet')
        ext_vars, ext_codes = self.make_schema_kwargs()
        schema_json = _jsonnet.evaluate_file(jsonnet_file,
                ext_vars={
                    'net_ip': self._netip,
                    'task': task,
                    'network': netw,
                    'dataset_name': dname,
                    'dataset_info': json.dumps(dinfo) if dinfo else '{}',
                    **ext_vars},
                ext_codes={
                    'levelid': str(levelid),
                    'debug': 'true' if self._debug else 'false',
                    'num_cpu': str(self._cpu_count), 'num_gpu': str(self._gpu_count),
                    **ext_codes})
        return 100000, json.dumps(json.loads(schema_json), separators=(',', ':'))

    def memstat(self, params):
        result = {
            **self.get_app_memstat(params),
            **k12ai_platform_memory_free()
        }
        return 100000, result

    def execute(self, token, op, user, uuid, params):
        Logger.info(f'{token}, {op}, {user}, {uuid}')
        container = self.get_container(user, uuid)
        phase, action = op.split('.')
        if action == 'stop':
            if container is None or container.status != 'running':
                return 100205, None

            Thread(target=lambda: self.stop_container_worker(
                token=token,
                op=op,
                user=user,
                uuid=uuid,
                container=container), daemon=True).start()
            return 100000, None

        if container:
            if container.status == 'running':
                return 100204, None
            container.remove()

        cons = self._docker.containers.list(filters={'label': 'k12ai.service.name'})
        if len(cons) > MAX_TASKS:
            return 100210, f'{len(cons) > MAX_TASKS}'

        if not isinstance(params, dict):
            return 100231, 'parameters type is not dict'

        Thread(target=lambda: self.start_container_worker(
            token=token,
            op=op,
            user=user,
            uuid=uuid,
            params=params), daemon=True).start()

        return 100000, None
