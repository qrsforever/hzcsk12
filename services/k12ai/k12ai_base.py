#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file base.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-03-02 00:31

import os
import json, _jsonnet
import docker
from threading import Thread

from k12ai.k12ai_utils import k12ai_utils_topdir
from k12ai.k12ai_errmsg import k12ai_error_message
from k12ai.k12ai_consul import k12ai_consul_message
from k12ai.k12ai_logger import Logger


class ServiceRPC(object):

    def __init__(self, sname, host, port, image, dataroot, debug):
        self._docker = docker.from_env()
        self._sname = sname
        self._image = image
        self._host = host
        self._port = port
        self._debug = debug
        self._workdir = f'/hzcsk12/{self._sname}'
        self._projdir = os.path.join(k12ai_utils_topdir(), self._sname)

        self._datadir = f'{dataroot}/datasets/{self._sname}'
        self._userdir = f'{dataroot}/users'
        self._commlib = os.path.join(k12ai_utils_topdir(), 'services', 'k12ai', 'common')
        self._jschema = os.path.join(self._projdir, 'app', 'templates', 'schema', f'k12ai_{self._sname}.jsonnet')

    def send_message(self, token, op, user, uuid, msgtype, message, clear=False):
        if not msgtype:
            return
        if isinstance(message, dict) and 'err_type' in message:
            errtype = message['err_type']
            errcode = self.errtype2errcode(errtype)
            if errcode < 0:
                if errtype == 'MemoryError':
                    errcode = 100901
                elif errtype == 'NotImplementedError':
                    errcode = 100902
                elif errtype == 'ConfigurationError':
                    errcode = 100903
                elif errtype == 'FileNotFoundError':
                    errcode = 100905
                else:
                    errcode = 100399
                message = k12ai_error_message(errcode, exc_info=message)
            else:
                message = k12ai_error_message(errcode, exc_info=message)
        k12ai_consul_message(token, user, op, f'k12{self._sname}', uuid, msgtype, message, clear)

    def errtype2errcode(self, errtype):
        raise NotImplementedError

    def make_container_command(self, op, cachedir, params):
        raise NotImplementedError

    def make_container_volumes(self):
        raise NotImplementedError

    def make_container_kwargs(self):
        raise NotImplementedError

    def make_schema_kwargs(self):
        raise NotImplementedError

    def get_container(self, user, uuid):
        try:
            cons = self._docker.containers.list(all=True, filters={'label': [
                'k12ai.service.user=%s'%user,
                'k12ai.service.uuid=%s'%uuid]})
            if len(cons) == 1:
                return cons[0]
        except docker.errors.NotFound:
            pass
        return None

    def get_cache_dir(self, user, uuid):
        usercache = '%s/%s/%s' % (self._userdir, user, uuid)
        if not os.path.exists(usercache):
            os.makedirs(usercache)
        return usercache

    def run_container(self, token, op, user, uuid, command):
        message = None
        labels = {
            'k12ai.service.name': f'k12{self._sname}',
            'k12ai.service.op': op,
            'k12ai.service.user': user,
            'k12ai.service.uuid': uuid
        }

        volumes = self.make_container_volumes()
        volumes[f'{self._datadir}'] = {'bind': f'/datasets', 'mode': 'rw'}
        volumes[f'{self._commlib}'] = {'bind': f'{self._workdir}/app/k12ai/common', 'mode': 'rw'}
        volumes[self.get_cache_dir(user, uuid)] = {'bind': f'/cache', 'mode': 'rw'}

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
            'detach': True,
            'runtime': 'nvidia',
            'labels': labels,
            'volumes': volumes,
            'environment': environs,
            **self.make_container_kwargs()
        }

        self.send_message(token, op, user, uuid, "status", {'value': 'starting'}, clear=True)
        try:

            self._docker.containers.run(f'{self._image}', command, **kwargs)
            return
        except Exception:
            message = k12ai_error_message(100203, f'container image:{self._image}', exc=True)
            self.send_message(token, op, user, uuid, "status", {'value': 'exit', 'way': 'docker'})
            self.send_message(token, op, user, uuid, "error", message)

    def schema(self, task, netw, dataset_name):
        Logger.info(f'{task}, {netw}, {dataset_name}')
        if not os.path.exists(self._jschema):
            return 100206, f'{self._jschema}'
        ext_vars, ext_codes = self.make_schema_kwargs()
        schema_json = _jsonnet.evaluate_file(self._jschema,
                ext_vars={
                    'task': task,
                    'network': netw,
                    'dataset_name': dataset_name,
                    **ext_vars},
                ext_codes={
                    'debug': 'true' if self._debug else 'false',
                    **ext_codes})
        return 100000, json.dumps(json.loads(schema_json), separators=(',', ':'))

    def execute(self, token, op, user, uuid, params):
        Logger.info(f'{token}, {op}, {user}, {uuid}')
        container = self.get_container(user, uuid)
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

        if not isinstance(params, dict):
            return 100231, 'parameters type is not dict'

        code, command = self.make_container_command(op, self.get_cache_dir(user, uuid), params)
        if code != 100000:
            return code, command

        Thread(target=lambda: self.run_container(token=token, op=op, user=user, uuid=uuid, command=command),
            daemon=True).start()
        return 100000, None
