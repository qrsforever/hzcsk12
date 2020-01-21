#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-15 16:38:16

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

service_name = 'k12nlp'

if os.environ.get("K12NLP_DEBUG"):
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

users_cache_dir = '/data/users'

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

def _check_config_diff(training_conf, serial_conf):
    if not os.path.exists(serial_conf):
        return True

    with open(training_conf, 'r') as f1:
        file1_params = json.loads(f1.read())
    with open(serial_conf, 'r') as f2:
        file2_params = json.loads(f2.read())

    diff = False
    for key in file1_params.keys() - file2_params.keys():
        logger.error(f"Key '{key}' found in training configuration but not in the serialization "
                     f"directory we're recovering from.")
        diff = True

    for key in file1_params.keys() - file2_params.keys():
        logger.error(f"Key '{key}' found in the serialization directory we're recovering from "
                     f"but not in the training config.")
        diff = True

    for key in file1_params.keys():
        if file1_params.get(key, None) != file2_params.get(key, None):
            logger.error(f"Value for '{key}' in training configuration does not match that the value in "
                         f"the serialization directory we're recovering from: "
                         f"{file1_params[key]} != {file2_params[key]}")
            diff = True
    return diff

OP_SUCCESS = 0
OP_FAILURE = -1

class NLPServiceRPC(object):

    def __init__(self,
            host, port,
            k12ai='k12ai',
            image='hzcsai_com/k12nlp',
            data_root='/data',
            workdir='/hzcsk12/nlp', debug=False):
        self._debug = debug
        self._host = host
        self._port = port
        self._k12ai = k12ai
        self._image = image
        self._docker = docker.from_env()
        self._workdir = workdir
        self._projdir = os.path.abspath(
                os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "..")
        logger.info('workdir:%s, projdir:%s', self._workdir, self._projdir)

        self.userscache_dir = '%s/users' % data_root
        self.datasets_dir = '%s/datasets/nlp' % data_root
        self.pretrained_dir = '%s/pretrained/nlp' % data_root
        self.nltk_data_dir = '%s/nltk_data' % data_root

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
                    code = 100405
                elif errtype == 'MemoryError':
                    code = 100901
                else:
                    code = 100499
                message = _err_msg(code, ext_info=message)

        data = {
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
        api = 'http://{}:{}/k12ai/private/message'.format(service['Address'], service['Port'])
        requests.post(api, json=data)
        if self._debug:
            if clear:
                client.kv.delete('framework/%s/%s' % (user, uuid), recurse=True)
            key = 'framework/%s/%s/%s/%s' % (user, uuid, op, msgtype)
            if msgtype != 'status':
                key = '%s/%s' % (key, data['datetime'][:-2])
            client.kv.put(key, json.dumps(data, indent=4))

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
        usercache = '%s/%s/%s'%(self.userscache_dir, user, uuid)
        if not os.path.exists(usercache):
            os.makedirs(usercache)
        return usercache

    def _prepare_environ(self, user, uuid, params):
        if not params or not isinstance(params, dict):
            return 100203, 'parameters type is not dict'

        resume = True
        if '_k12.data.dataset_name' in params.keys():
            config_tree = ConfigFactory.from_dict(params)
            _k12ai_tree = config_tree.pop('_k12')
            resume = _k12ai_tree.get('model.resume', True)
            config_str = HOCONConverter.convert(config_tree, 'json')
        else:
            config_str = json.dumps(params)

        config_file = '%s/config.json' % self._get_cache_dir(user, uuid)
        with open(config_file, 'w') as fout:
            fout.write(config_str)

        return 100000, {'resume': resume}

    def _run(self, op, user, uuid, command=None):
        logger.info(command)
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
                self.datasets_dir: {'bind': '/datasets', 'mode': 'ro'},
                usercache_dir: {'bind':'/cache', 'mode': 'rw'},
                self.pretrained_dir: {'bind': '/pretrained', 'mode': 'ro'},
                self.nltk_data_dir: {'bind': '/root/nltk_data', 'mode': 'ro'},
                }

        if self._debug:
            rm_flag = False
            volumes['%s/app'%self._projdir] = {'bind':'%s/app'%self._workdir, 'mode':'rw'}
            volumes['%s/allennlp/allennlp'%self._projdir] = {'bind':'%s/allennlp'%self._workdir, 'mode':'rw'}
            volumes['%s/allennlp-reading-comprehension/allennlp_rc'%self._projdir] = {'bind':'%s/allennlp_rc'%self._workdir, 'mode':'rw'}

        environs = {
                'K12NLP_RPC_HOST': '%s' % self._host,
                'K12NLP_RPC_PORT': '%s' % self._port,
                'K12NLP_OP': '%s' % op,
                'K12NLP_USER': '%s' % user,
                'K12NLP_UUID': '%s' % uuid
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

        self.send_message(op, user, uuid, "status", {'value':'starting'}, clear=True)
        try:
            self._docker.containers.run(self._image, command, **kwargs)
            return
        except Exception:
            message = _err_msg(100402, 'container image:{}'.format(self._image), exc=True)
            self.send_message(op, user, uuid, "status", {'value':'exit', 'way': 'docker'})

        if message:
            self.send_message(op, user, uuid, "error", message)

    def schema(self, task, netw, dataset_name):
        schema_file = os.path.join(self._projdir, 'app', 'templates', 'schema', 'k12ai_nlp.jsonnet')
        if not os.path.exists(schema_file):
            return OP_FAILURE, f'{schema_file}'
        schema_json = _jsonnet.evaluate_file(schema_file, ext_vars={
            'task': task, 
            'network': netw,
            'dataset_name': dataset_name})
        return 100000, json.dumps(json.loads(schema_json), separators=(',',':'))
    
    def execute(self, op, user, uuid, params):
        logger.info("call execute(%s, %s, %s)", op, user, uuid)
        container = self._get_container(user, uuid)
        phase, action = op.split('.')
        if action == 'stop':
            if container is None or container.status != 'running':
                return 100205, None
            container.kill()
            self.send_message('%s.start' % phase, user, uuid, "status", {'value':'exit', 'way': 'manual'})
            return 100000, None

        if container:
            if container.status == 'running':
                return 100204, None
            container.remove()

        code, result = self._prepare_environ(user, uuid, params)
        if code != 100000:
            return code, result

        if phase == 'train':
            flag = '--force'
            if result['resume']:
                config_conf = os.path.join(self._get_cache_dir(user, uuid), 'config.json')
                serial_conf = os.path.join(self._get_cache_dir(user, uuid), 'output', 'config.json')
                if os.path.exists(serial_conf):
                    if not _check_config_diff(config_conf, serial_conf):
                        flag = '--recover'
            command = 'allennlp train /cache/config.json %s --serialization-dir /cache/output' % flag
        elif phase == 'evaluate':
            raise('not impl yet')
        elif phase == 'predict':
            raise('not impl yet')

        command += ' --include-package allennlp_rc'
        Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return 100000, None

    # def evaluate(self, op, user, uuid, params):
    #     logger.info("call evaluate(%s, %s, %s)", op, user, uuid)
    #     if op == 'evaluate.stop':
    #         Thread(target=lambda: self._run(op=op, user=user, uuid=uuid),
    #                 daemon=True).start()
    #         return OP_SUCCESS, None

    #     if not params or not isinstance(params, dict):
    #         return OP_FAILURE, 'parameter is none or not dict type'

    #     input_file = params.get('input_file', None)
    #     if not input_file:
    #         return OP_FAILURE, 'parameter have no key: input_file'

    #     pro_dir = os.path.join(users_cache_dir, user, uuid)
    #     archive_file = os.path.join(pro_dir, 'output', 'model.tar.gz')
    #     if not os.path.exists(archive_file):
    #         return OP_FAILURE, f'model.tar.gz is not found in {pro_dir}'
    #     output_file = params.get('output_file', None)
    #     if not output_file:
    #         output_file = os.path.join(pro_dir, 'evaluate_output.txt')

    #     command = 'allennlp evaluate {} {} --output-file {}'.format(
    #             archive_file, input_file, output_file)
    #     Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
    #             daemon=True).start()
    #     return OP_SUCCESS, None

    # def predict(self, op, user, uuid, params):
    #     logger.info("call predict(%s, %s, %s)", op, user, uuid)
    #     if op == 'predict.stop':
    #         Thread(target=lambda: self._run(op=op, user=user, uuid=uuid),
    #                 daemon=True).start()
    #         return OP_SUCCESS, None

    #     if not params or not isinstance(params, dict):
    #         return OP_FAILURE, 'params is none or not dict type'

    #     pro_dir = os.path.join(users_cache_dir, user, uuid)
    #     archive_file = os.path.join(pro_dir, 'output', 'model.tar.gz')
    #     if not os.path.exists(archive_file):
    #         return OP_FAILURE, f'model.tar.gz is not found in {pro_dir}'

    #     input_type = params.get('input_type', None)
    #     if not input_type:
    #         return OP_FAILURE, 'parameter have no key: input_file'

    #     if input_type == 'text':
    #         input_text = params.get('input_text', None)
    #         if not input_text:
    #             return OP_FAILURE, 'parameter have no key: input_file'
    #         input_file = os.path.join(pro_dir, 'predict_input.txt')
    #         with open(input_file, 'w') as fout:
    #             fout.write(input_text)

    #     output_file = params.get('output_file', None)
    #     if not output_file:
    #         output_file = os.path.join(pro_dir, 'predict_output.json')

    #     other_args = ''
    #     batch_size = params.get('batch_size', None)
    #     if batch_size and isinstance(batch_size, int):
    #         other_args += ' --batch-size %d' % batch_size
    #     predictor = params.get('predictor', None)
    #     if predictor:
    #         other_args += ' --predictor %s' % predictor

    #     command = 'allennlp predict {} {} --output-file {} {}'.format(
    #             archive_file, input_file, output_file, other_args)
    #     Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
    #             daemon=True).start()
    #     return OP_SUCCESS, None

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
            default=8149,
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

    host = args.host if args.host else app_host_ip

    consul_addr = args.consul_addr if args.consul_addr else app_host_ip
    consul_port = args.consul_port

    thread = Thread(target=_delay_do_consul, args=(host, args.port))
    thread.start()

    logger.info('start zerorpc server on %s:%d', host, args.port)

    try:
        app = zerorpc.Server(NLPServiceRPC(
            host=host, port=args.port,
            k12ai='{}-k12ai'.format(app_host_name),
            image=args.image,
            data_root=args.data_root,
            debug=LEVEL == logging.DEBUG))
        app.bind('tcp://%s:%d' % (host, args.port))
        app.run()
    finally:
        app_quit = True
        thread.join()
