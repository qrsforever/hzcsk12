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

try:
    from k12ai_errmsg import k12ai_error_message as _err_msg
    from k12ai_utils import k12ai_get_hostname as _get_hostname
    from k12ai_utils import k12ai_get_hostip as _get_hostip
except Exception:
    topdir = os.path.abspath(
            os.path.dirname(os.path.abspath(__file__)) + "/../..")
    sys.path.append(topdir)
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
            workdir='/hzcsk12/nlp', debug=False):
        self._debug = debug
        self._host = host
        self._port = port
        self._k12ai = k12ai
        self._image = image
        self._libs = 'app/k12nlp' 
        self._docker = docker.from_env()
        self._workdir = workdir
        self._projdir = os.path.abspath(
                os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "..")
        logger.info('workdir:%s, projdir:%s', self._workdir, self._projdir)

    def send_message(self, op, user, uuid, msgtype, message):
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
        data[msgtype] = message

        # service
        api = 'http://{}:{}/k12ai/private/message'.format(service['Address'], service['Port'])
        requests.post(api, json=data)
        if self._debug:
            key = 'framework/%s/%s/%s/%s' % (user, uuid, op, msgtype)
            if msgtype != 'status':
                key = '%s/%s' % (key, data['datetime'][:-2])
            client.kv.put(key, json.dumps(data, indent=4))

    def schema(self, service_task, dataset_path, dataset_name):
        schema_file = os.path.join(self._projdir, 'app', 'templates', 'schema', 'k12ai_nlp.jsonnet')
        if not os.path.exists(schema_file):
            return OP_FAILURE, f'schema file: {schema_file} not found'
        schema_json = _jsonnet.evaluate_file(schema_file, ext_vars={
            'task': service_task, 
            'dataset_path': dataset_path,
            'dataset_name': dataset_name})
        return OP_SUCCESS, schema_json

    def _get_container(self, op, user, uuid):
        container_name = '%s-%s-%s' % (op.split('.')[0], user, uuid)
        try:
            cons = self._docker.containers.list(all=True, filters={'label': [
                'k12ai.service.user=%s'%user,
                'k12ai.service.uuid=%s'%uuid]})
            if len(cons) == 1:
                return container_name, cons[0]
        except docker.errors.NotFound:
            pass
        return container_name, None

    def _run(self, op, user, uuid, command=None):
        logger.info(command)
        message = None
        container_name, con = self._get_container(op, user, uuid)

        if not command: # stop
            try:
                if con:
                    if con.status == 'running':
                        con.kill()
                        message = _err_msg(100400, f'container name:{container_name}')
                        xop = op.replace('stop', 'start')
                        self.send_message(xop, user, uuid, "status", {'value':'exit', 'way': 'manual'})
                else:
                    message = _err_msg(100401, f'container name:{container_name}')
            except Exception:
                message = _err_msg(100403, f'container name:{container_name}', exc=True)
        else: # start
            rm_flag = True
            labels = {
                    'k12ai.service.name': service_name,
                    'k12ai.service.op': op,
                    'k12ai.service.user': user,
                    'k12ai.service.uuid': uuid
                    }
            volumes = {
                    '/data': {'bind':'/data', 'mode':'rw'}
                    }
            if self._debug:
                rm_flag = False
                volumes['%s/app'%self._projdir] = {'bind':'%s/app'%self._workdir, 'mode':'rw'}
                volumes['%s/allennlp'%self._projdir] = {'bind':'%s/allennlp'%self._workdir, 'mode':'rw'}
            environs = {
                    'K12NLP_RPC_HOST': '%s' % self._host,
                    'K12NLP_RPC_PORT': '%s' % self._port,
                    'K12NLP_OP': '%s' % op,
                    'K12NLP_USER': '%s' % user,
                    'K12NLP_UUID': '%s' % uuid
                    }
            kwargs = {
                    'name': container_name,
                    'auto_remove': rm_flag,
                    'detach': True,
                    'runtime': 'nvidia',
                    'labels': labels,
                    'volumes': volumes,
                    'environment': environs
                    }

            if con and con.status == 'running':
                message = _err_msg(100404, 'container name: {}'.format(con.short_id))
            else:
                self.send_message(op, user, uuid, "status", {'value':'starting'})
                try:
                    if con:
                        print("#################")
                        con.remove()
                    con = self._docker.containers.run(self._image,
                            command, **kwargs)
                    return
                except Exception:
                    message = _err_msg(100402, 'container image:{}'.format(self._image), exc=True)
                    self.send_message(op, user, uuid, "status", {'value':'exit', 'way': 'docker'})

        if message:
            self.send_message(op, user, uuid, "error", message)

    def train(self, op, user, uuid, params):
        logger.info("call train(%s, %s, %s)", op, user, uuid)
        if op == 'train.stop':
            Thread(target=lambda: self._run(op=op, user=user, uuid=uuid),
                    daemon=True).start()
            return OP_SUCCESS, None

        container_name, con = self._get_container(op, user, uuid)
        if con and con.status == 'running':
            return OP_FAILURE, f'[{container_name}] same op is running!'

        if not params or not isinstance(params, dict):
            return OP_FAILURE, 'parameter is none or not dict type'

        pro_dir = os.path.join(users_cache_dir, user, uuid)
        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)
        config_tree = ConfigFactory.from_dict(params)
        config_tree.pop('_k12')
        config_str = HOCONConverter.convert(config_tree, 'json')
        training_config = os.path.join(pro_dir, 'config.json')
        with open(training_config, 'w') as fout:
            fout.write(config_str)
        flag = '--force'
        output_dir = os.path.join(pro_dir, 'output')
        serial_conf = os.path.join(output_dir, 'config.json')
        if os.path.exists(serial_conf):
            if not _check_config_diff(training_config, serial_conf):
                flag = '--recover'
        # command = 'allennlp train {} {} -s {} --include-package {}'.format(
                # training_config, flag, output_dir, self._libs)
        command = 'allennlp train {} {} -s {}'.format(
                training_config, flag, output_dir)
        Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return OP_SUCCESS, None

    def evaluate(self, op, user, uuid, params):
        logger.info("call evaluate(%s, %s, %s)", op, user, uuid)
        if op == 'evaluate.stop':
            Thread(target=lambda: self._run(op=op, user=user, uuid=uuid),
                    daemon=True).start()
            return OP_SUCCESS, None

        if not params or not isinstance(params, dict):
            return OP_FAILURE, 'parameter is none or not dict type'

        input_file = params.get('input_file', None)
        if not input_file:
            return OP_FAILURE, 'parameter have no key: input_file'

        pro_dir = os.path.join(users_cache_dir, user, uuid)
        archive_file = os.path.join(pro_dir, 'output', 'model.tar.gz')
        if not os.path.exists(archive_file):
            return OP_FAILURE, f'model.tar.gz is not found in {pro_dir}'
        output_file = params.get('output_file', None)
        if not output_file:
            output_file = os.path.join(pro_dir, 'evaluate_output.txt')

        command = 'allennlp evaluate {} {} --output-file {} --include-package {}'.format(
                archive_file, input_file, output_file, self._libs)
        Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return OP_SUCCESS, None

    def predict(self, op, user, uuid, params):
        logger.info("call predict(%s, %s, %s)", op, user, uuid)
        if op == 'predict.stop':
            Thread(target=lambda: self._run(op=op, user=user, uuid=uuid),
                    daemon=True).start()
            return OP_SUCCESS, None

        if not params or not isinstance(params, dict):
            return OP_FAILURE, 'params is none or not dict type'

        pro_dir = os.path.join(users_cache_dir, user, uuid)
        archive_file = os.path.join(pro_dir, 'output', 'model.tar.gz')
        if not os.path.exists(archive_file):
            return OP_FAILURE, f'model.tar.gz is not found in {pro_dir}'

        input_type = params.get('input_type', None)
        if not input_type:
            return OP_FAILURE, 'parameter have no key: input_file'

        if input_type == 'text':
            input_text = params.get('input_text', None)
            if not input_text:
                return OP_FAILURE, 'parameter have no key: input_file'
            input_file = os.path.join(pro_dir, 'predict_input.txt')
            with open(input_file, 'w') as fout:
                fout.write(input_text)

        output_file = params.get('output_file', None)
        if not output_file:
            output_file = os.path.join(pro_dir, 'predict_output.json')

        other_args = ''
        batch_size = params.get('batch_size', None)
        if batch_size and isinstance(batch_size, int):
            other_args += ' --batch-size %d' % batch_size
        predictor = params.get('predictor', None)
        if predictor:
            other_args += ' --predictor %s' % predictor

        command = 'allennlp predict {} {} --output-file {} --include-package {} {}'.format(
                archive_file, input_file, output_file, self._libs, other_args)
        Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return OP_SUCCESS, None

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
            default=None,
            type=str,
            dest='image',
            help="image to run container")
    args = parser.parse_args()

    image = args.image if args.image else 'hzcsai_com/k12nlp-dev'

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
            image=image, debug=LEVEL == logging.DEBUG))
        app.bind('tcp://%s:%d' % (host, args.port))
        app.run()
    finally:
        app_quit = True
        thread.join()
