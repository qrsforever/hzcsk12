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
from threading import Thread

from k12ai.k12ai_utils import (k12ai_utils_topdir, k12ai_utils_netip)
from k12ai.k12ai_errmsg import (FrameworkError, k12ai_error_message, gen_exc_info)
from k12ai.k12ai_consul import k12ai_consul_message
from k12ai.k12ai_logger import Logger
from k12ai.k12ai_platform import ( # noqa
        k12ai_platform_cpu_count,
        k12ai_platform_gpu_count,
        k12ai_platform_stats,
        k12ai_platform_memory_free)
from k12ai.k12ai_utils import ( # noqa
        k12ai_oss_client, k12ai_object_remove,
        k12ai_object_get, k12ai_object_put)

MAX_TASKS = 10
MAX_MSIZE = 2000 # max model size


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

        self._userdir = f'{dataroot}/users'
        self._commlib = os.path.join(k12ai_utils_topdir(), 'services', 'k12ai', 'common')
        self._jschema = os.path.join(self._projdir, 'app', 'templates', 'schema')

    def send_message(self, appId, token, op, user, uuid, msgtype, message, clear=False):
        if not msgtype:
            return
        if msgtype in ('error', 'runlog'):
            errcode = 000000 if msgtype == 'runlog' else 999999
            if 'errinfo' in message:
                if 'warning' == message['status']:
                    errcode = 100009
                elif isinstance(message['errinfo'], dict):
                    errcode, message = self.container_on_crash(appId, op, user, uuid, message)
                Logger.info(message)
            elif 'status' in message:
                if 'starting' == message['status']:
                    errcode = 100001
                elif 'running' == message['status']:
                    errcode = 100002
                elif message['status'] in ('stopped', 'paused', 'robbed'):
                    errcode, message = self.container_on_stop(appId, op, user, uuid, message)
                elif 'finished' == message['status']:
                    errcode, message = self.container_on_finished(appId, op, user, uuid, message)
                elif 'monitor' == message['status']:
                    errcode, message = self.container_on_monitor(appId, op, user, uuid, message)
                    if errcode < 0:
                        return
            message = k12ai_error_message(errcode, expand=message)

        if isinstance(message, dict):
            if len(json.dumps(message)) > 500000: # max 512kb
                errcode = 100011
                message = k12ai_error_message(errcode)
        k12ai_consul_message(f'k12{self._sname}', appId, token, op, user, uuid, msgtype, message, clear)

    def container_on_crash(self, appId, op, user, uuid, message):
        message = self.on_finished(appId, op, user, uuid, message)
        if 'err_code' in message['errinfo']:
            return message['errinfo']['err_code'], message

        errinfo = message['errinfo']
        if 'err_type' in errinfo and 'err_text' in errinfo:
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
                    if 'CUDA out of memory' in errtext or 'CUDA error: out of memory' in errtext:
                        errcode = 100908
                elif errtype == 'ImageNotFound':
                    errcode = 100905
        else:
            errcode = 999999
        return errcode, message

    def container_on_stop(self, appId, op, user, uuid, message):
        message = self.on_finished(appId, op, user, uuid, message)
        kStatuCodes = {
            'stopped': 100004,
            'paused':  100005, # noqa
            'robbed':  100006  # noqa
        }
        return kStatuCodes[message['status']], message

    def container_on_finished(self, appId, op, user, uuid, message):
        message = self.on_finished(appId, op, user, uuid, message)
        return 100003, message

    def container_on_monitor(self, appId, op, user, uuid, message):
        if isinstance(message, dict) and message['monitor'] == 'gpu_memory':
            code, resource = k12ai_platform_stats(appId, 'query', user, uuid, {'services':True}, isasync=False)
            if code == 100000 and len(resource['services']) == 1:
                info = resource['services'][0]
                if 'service_gpu_memory_usage_MB' in info:
                    if info['service_gpu_memory_usage_MB'] > MAX_MSIZE:
                        return 100012, info
        return -1, None

    def pre_processing(self, appId, op, user, uuid, params):
        return params

    def post_processing(self, appId, op, user, uuid, message):
        return message

    def on_starting(self, appId, op, user, uuid, params):
        self.clear_cache(user, uuid)
        return self.pre_processing(appId, op, user, uuid, params)

    def on_finished(self, appId, op, user, uuid, message):
        self.post_processing(appId, op, user, uuid, message)
        self.clear_cache(user, uuid)
        return message

    def oss_upload(self, filepath, bucket_name=None, prefix_map=None, clear=False):
        if not os.path.exists(filepath):
            return
        try:
            if clear:
                k12ai_object_remove(self._ossmc, remote_path=filepath)
            result = k12ai_object_put(self._ossmc, local_path=filepath,
                    bucket_name=bucket_name, prefix_map=prefix_map)
            Logger.info(result)
            return result
        except Exception as err:
            Logger.error(str(err))
            return {}

    def oss_download(self, filepath, bucket_name=None, prefix_map=None):
        try:
            result = k12ai_object_get(self._ossmc, remote_path=filepath,
                    bucket_name=bucket_name, prefix_map=prefix_map)
            Logger.info(result)
        except Exception as err:
            Logger.error(str(err))

    def oss_remove(self, filepath, bucket_name=None):
        try:
            k12ai_object_remove(self._ossmc, remote_path=filepath)
        except Exception as err:
            Logger.error(str(err))

    def make_container_command(self, appId, op, cachedir, params):
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

    def start_container_worker(self, appId, token, op, user, uuid, params):
        usercache, innercache = self.get_cache_dir(user, uuid)
        try:
            params = self.on_starting(appId, op, user, uuid, params)

            dev = False
            if 'developer' in params.keys():
                dev = params['developer']

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
                self._commlib: {'bind': f'{self._workdir}/app/k12ai/common', 'mode': 'ro'},
                **self.make_container_volumes()
            }

            if tb_logdir:
                volumes['/data/tblogs'] = {'bind': '/data/tblogs', 'mode':'rw'}

            environs = {
                'K12AI_RPC_HOST': self._host,
                'K12AI_RPC_PORT': self._port,
                'K12AI_APPID': appId,
                'K12AI_TOKEN': token,
                'K12AI_OP': op,
                'K12AI_USER': user,
                'K12AI_UUID': uuid,
                **self.make_container_environs(op, params)
            }

            if dev:
                environs['K12AI_DEVELOPER'] = dev

            if tb_logdir:
                environs['K12AI_TBLOG_DIR'] = tb_logdir

            # W: don't set hostname
            kwargs = {
                'name': '%s-%s-%s' % (op, user, uuid),
                'auto_remove': not dev,
                'detach': True,
                'runtime': 'nvidia',
                'labels': labels,
                'volumes': volumes,
                'environment': environs,
                **self.make_container_kwargs(op, params)
            }

            self.send_message(appId, token, op, user, uuid, "error", {'status': 'starting'}, clear=True)
            command = self.make_container_command(appId, op, user, uuid, params)
            Logger.info(kwargs)
            self._docker.containers.run(f'{self._image}', command, **kwargs)
            return
        except FrameworkError as fwerr:
            self.send_message(appId, token, op, user, uuid, "error", {
                'status': 'crash', 'errinfo': {'err_code': fwerr.errcode, 'err_text': fwerr.errtext}
            })
        except Exception:
            self.send_message(appId, token, op, user, uuid, "error", {
                'status': 'crash', 'errinfo': gen_exc_info()
            })

    def stop_container_worker(self, appId, token, status, user, uuid, container):
        cop = container.attrs['Config']['Labels']['k12ai.service.op']
        try:
            container.kill()
            self.send_message(appId, token, cop, user, uuid, "error", {'status': status})
        except Exception as err:
            Logger.error(str(err))
            self.send_message(appId, token, cop, user, uuid, "error", {
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

    def execute(self, appId, token, op, user, uuid, params):
        Logger.info(f'appId:{appId}, token:{token}, op:{op}, user:{user}, uuid:{uuid}')
        container = self.get_container(user, uuid)
        phase, action = op.split('.')
        if action in ('stop', 'pause', 'rob'):
            if container is None or container.status != 'running':
                return 100205, None

            status = 'stopped'
            if action == 'pause':
                status = 'paused'
            elif action == 'rob':
                status = 'robbed'

            Thread(target=lambda: self.stop_container_worker(
                appId=appId,
                token=token,
                status=status,
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

        if 'train.start' == op and not isinstance(params, dict):
            return 100231, 'parameters type is not dict'

        Thread(target=lambda: self.start_container_worker(
            appId=appId,
            token=token,
            op=op,
            user=user,
            uuid=uuid,
            params=params), daemon=True).start()

        return 100000, None
