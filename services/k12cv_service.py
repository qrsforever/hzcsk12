#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12cv_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-27 17:08:18

import os, sys, time
import argparse
import logging, json
import _jsonnet
import zerorpc
import requests
import consul
import docker
from threading import Thread
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter

from k12ai_errmsg import k12ai_error_message as _err_msg
from k12ai_utils import k12ai_get_hostname as _get_hostname
from k12ai_utils import k12ai_get_hostip as _get_hostip

service_name = 'k12cv'

if os.environ.get("K12CV_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

logger = logging.getLogger(__name__)

app_quit = False
app_host_name = _get_hostname()
app_host_ip = _get_hostip()

consul_addr = None
consul_port = None

pretrained_models = {
    'vgg11': 'vgg11-bbd30ac9.pth',
    'vgg13': 'vgg13-c768596a.pth',
    'vgg16': 'vgg16-397923af.pth',
    'vgg19': 'vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'vgg11_bn-6002323d.pth',
    'vgg13_bn': 'vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'vgg19_bn-c79401a0.pth',
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def _delay_do_consul(host, port):
    time.sleep(3)
    while not app_quit:
        try:
            client = consul.Consul(host=consul_addr, port=consul_port)
            client.agent.service.register(
                name='{}-{}'.format(app_host_name, service_name),
                address=host, port=port, tags=('AI', 'FRAMEWORK'),
                check=consul.Check.tcp(host, port,
                    interval='10s', timeout='5s', deregister='10s'))
            break
        except Exception as err:
            logger.info("consul agent service register err", err)
            time.sleep(3)


OP_FAILURE = -1


class CVServiceRPC(object):

    def __init__(self,
            host, port,
            k12ai='k12ai',
            image='hzcsai_com/k12cv',
            data_root='/data',
            workdir='/hzcsk12/cv', debug=False):
        self._debug = debug
        self._host = host
        self._port = port
        self._k12ai = k12ai
        self._image = image
        self._docker = docker.from_env()
        self._workdir = workdir
        self._projdir = os.path.abspath( # noqa: E126
                os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "..")
        logger.info('workdir:%s, projdir:%s', self._workdir, self._projdir)

        self.userscache_dir = '%s/users' % data_root
        self.datasets_dir = '%s/datasets/cv' % data_root
        self.pretrained_dir = '%s/pretrained/cv' % data_root

    def send_message(self, op, user, uuid, msgtype, message, clear=False):
        client = consul.Consul(consul_addr, port=consul_port)
        service = client.agent.services().get(self._k12ai)
        if not service:
            logger.error("Not found %s service!" % self._k12ai)
            return
        if not msgtype:
            return
        if isinstance(message, dict):
            if 'err_type' in message:
                errtype = message['err_type']
                if errtype == 'ConfigurationError':
                    code = 100305
                elif errtype == 'ImageTypeError':
                    code = 100306
                elif errtype == 'TensorSizeError':
                    code = 100307
                elif errtype == 'MemoryError':
                    code = 100901
                else:
                    code = 100399
                message = _err_msg(code, ext_info=message)

        data = { # noqa: E126
                'version': '0.1.0',
                'type': msgtype,
                'tag': 'framework',
                'op': op,
                'user': user,
                'service_uuid': uuid,
                }
        now_time = time.time()
        data['timestamp'] = round(now_time * 1000)
        data['datetime'] = time.strftime('%Y%m%d%H%M%S', time.localtime(now_time))
        data['data'] = message

        # service
        api = 'http://{}:{}/k12ai/private/message?type={}'.format(service['Address'], service['Port'], msgtype)
        requests.post(api, json=data)
        if self._debug:
            if clear:
                client.kv.delete('framework/%s/%s' % (user, uuid), recurse=True)
            key = 'framework/%s/%s/%s/%s' % (user, uuid, op, msgtype)
            if msgtype != 'status':
                key = '%s/%s' % (key, data['datetime'][:-2])
            client.kv.put(key, json.dumps(data, indent=2))

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
        if not params or not isinstance(params, dict):
            return 100203, 'parameters type is not dict'

        if '_k12.data.dataset_name' in params.keys():
            config_tree = ConfigFactory.from_dict(params)
            _k12ai_tree = config_tree.pop('_k12')

            # Aug Trans
            if config_tree.get('train.aug_trans.trans_seq', default=None) is None:
                config_tree.put('train.aug_trans.trans_seq', [])
            if config_tree.get('val.aug_trans.trans_seq', default=None) is None:
                config_tree.put('val.aug_trans.trans_seq', [])
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

            # CheckPoints
            model_name = config_tree.get('network.model_name', default='unknow')
            backbone = config_tree.get('network.backbone', default='unknow')
            ckpts_name = '%s_%s_%s' % (model_name, backbone, _k12ai_tree.get('data.dataset_name'))
            config_tree.put('network.checkpoints_root', '/cache')
            config_tree.put('network.checkpoints_name', ckpts_name)
            config_tree.put('network.checkpoints_dir', 'ckpts')

            if config_tree.get('network.resume_continue', default=False):
                resume_path = '%s/ckpts/%s_latest.pth' % (self._get_cache_dir(user, uuid), ckpts_name)
                if os.path.exists(resume_path):
                    config_tree.put('network.resume', '/cache/ckpts/%s_latest.pth' % ckpts_name)

            # Pretrained
            pretrained = config_tree.get('network.pretrained', default=False)
            config_tree.pop('network.pretrained', default=None)
            if pretrained:
                _file = pretrained_models.get(backbone, 'nofile')
                if os.path.exists('%s/%s' % (self.pretrained_dir, _file)):
                    config_tree.put('network.pretrained', '/pretrained/%s' % _file)

            config_str = HOCONConverter.convert(config_tree, 'json')
        else:
            config_str = json.dumps(params)

        config_file = '%s/config.json' % self._get_cache_dir(user, uuid)
        with open(config_file, 'w') as fout:
            fout.write(config_str)

        return 100000, None

    def _run(self, op, user, uuid, command=None):
        logger.info(command)
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
            message = _err_msg(100302, 'container image:{}'.format(self._image), exc=True)
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
        logger.info("call execute(%s, %s, %s)", op, user, uuid)
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

        code, result = self._prepare_environ(user, uuid, params)
        if code != 100000:
            return code, result

        command = 'python -m torch.distributed.launch --nproc_per_node=1 {}'.format(
                '%s/torchcv/main.py' % self._workdir)

        command += ' --config_file /cache/config.json'

        if phase == 'train':
            command += ' --phase train'
        elif phase == 'evaluate':
            command += ' --phase test --out_dir /cache/output'
        elif phase == 'predict':
            raise('not impl yet')

        Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
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

    host = args.host if args.host else app_host_ip

    consul_addr = args.consul_addr if args.consul_addr else app_host_ip
    consul_port = args.consul_port

    thread = Thread(target=_delay_do_consul, args=(host, args.port))
    thread.start()

    logger.info('start zerorpc server on %s:%d', host, args.port)

    try:
        app = zerorpc.Server(CVServiceRPC(
            host=host, port=args.port,
            k12ai='{}-k12ai'.format(app_host_name),
            image=args.image,
            data_root=args.data_root,
            debug=LEVEL==logging.DEBUG)) # noqa
        app.bind('tcp://%s:%d' % (host, args.port))
        app.run()
    finally:
        app_quit = True
        thread.join()
