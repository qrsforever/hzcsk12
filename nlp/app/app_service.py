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
import socket
import zerorpc
import requests
import consul
import docker
import traceback
from threading import Thread

try:
    from k12ai_errmsg import hzcsk12_error_message as _err_msg
except:
    try:
        topdir = os.path.abspath(
                os.path.dirname(os.path.abspath(__file__)) + "/../..")
        sys.path.append(topdir)
        from k12ai_errmsg import hzcsk12_error_message as _err_msg
    except:
        def _err_msg(code, message=None, detail=None, exc=None, exc_info=None):
            return {'code': 999999}

service_name = 'k12nlp'

if os.environ.get("K12NLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

logger = logging.getLogger(__name__)

app_quit = False

def _get_host_name():
    val = os.environ.get('HOST_NAME', None)
    if val:
        return val
    else:
        return socket.gethostname()

def _get_host_ip():
    val = os.environ.get('HOST_ADDR', None)
    if val:
        return val
    else:
        try:
            s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            s.connect(('8.8.8.8',80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip

app_host_name = _get_host_name()
app_host_ip = _get_host_ip()

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
            image='hzcsai_com/k12nlp-dev',
            libs='hzcsnlp',
            workdir='/hzcsk12', debug=False):
        self._debug = debug
        self._host = host
        self._port = port
        self._k12ai = k12ai
        self._image = image
        self._libs = libs
        self._docker = docker.from_env()
        if debug:
            self._workdir = workdir
            self._projdir = os.path.abspath(
                    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "..")
            logger.info('workdir:%s, projdir:%s', self._workdir, self._projdir)

    def send_message(self, task, user, uuid, msgtype, message):
        client = consul.Consul(consul_addr, port=consul_port)
        service = client.agent.services().get(self._k12ai)
        if not service:
            logger.error("Not found %s service!" % self._k12ai)
            return

        if msgtype == 'except' and isinstance(message, dict):
            cls = message['exc_type']
            if cls == 'ConfigurationError':
                code = 100405
            elif cls == 'MemoryError':
                code = 100901
            else:
                code = 100499
            message = _err_msg(code, exc_info=message)
            msgtype = 'error'

        data = {
                'version': '0.1.0',
                'type': msgtype,
                'tag': 'framework',
                'op': task,
                'user': user,
                'service_uuid': uuid,
                }
        now_time = time.time()
        data['timestamp'] = round(now_time * 1000)
        data['datetime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_time))
        data[msgtype] = message

        # service
        api = 'http://{}:{}/k12ai/private/message'.format(service['Address'], service['Port'])
        requests.post(api, json=data)
        if self._debug:
            client.kv.put('framework/%s/%s/%s/%s'%(user, uuid, task, msgtype), json.dumps(data, indent=4))

    def _run(self, task, user, uuid, command=None):
        message = {}
        stopcmd = command == None
        container_name = '%s-%s-%s' % (task.split('.')[0], user, uuid)
        try:
            con = self._docker.containers.get(container_name)
        except docker.errors.NotFound as err1:
            con = None

        if stopcmd: # stop
            try:
                if con:
                    if con.status == 'running':
                        con.stop()
                        message = _err_msg(100400, f'container name:{container_name}')
                else:
                    message = _err_msg(100401, f'container name:{container_name}')
            except Exception as err:
                message = _err_msg(100403, f'container name:{container_name}', exc=True)
        else: # start
            rm_flag = True
            labels = {
                    'k12ai.service.name': service_name
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
                    'K12NLP_TASK': '%s' % task,
                    'K12NLP_USER': '%s' % user,
                    'K12NLP_UUID': '%s' % uuid
                    }
            kwargs = {
                    'name': container_name,
                    'auto_remove': rm_flag,
                    'detach': True,
                    'network_mode': 'host',
                    'runtime': 'nvidia',
                    'shm_size': '2g',
                    'labels': labels,
                    'volumes': volumes,
                    'environment': environs
                    }
            try:
                if not con or con.status != 'running':
                    if con:
                        con.remove()
                    con = self._docker.containers.run(self._image,
                            command, **kwargs)
                else:
                    if con:
                        message = _err_msg(100404, 'container name: {}'.format(con.short_id))
            except Exception as err:
                message = _err_msg(100402, 'container image:{}'.format(self._image), exc=True)

        self.send_message(task, user, uuid, "error", message)

    def train(self, op, user, uuid, params):
        logger.info("call train(%s, %s, %s)", op, user, uuid)
        if op == 'train.stop':
            Thread(target=lambda: self._run(task=op, user=user, uuid=uuid),
                    daemon=True).start()
            return OP_SUCCESS, None

        if not params or not isinstance(params, dict):
            return OP_FAILURE, 'parameter is none or not dict type'

        pro_dir = os.path.join(users_cache_dir, user, uuid)
        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)
        training_config = os.path.join(pro_dir, 'config.json')
        with open(training_config, 'w') as fout:
            fout.write(json.dumps(params))
        flag = '--force'
        output_dir = os.path.join(pro_dir, 'output')
        serial_conf = os.path.join(output_dir, 'config.json')
        if os.path.exists(serial_conf):
            if not _check_config_diff(training_config, serial_conf):
                flag = '--recover'
        command = 'allennlp train {} {} -s {} --include-package {}'.format(
                training_config, flag, output_dir, self._libs)
        Thread(target=lambda: self._run(task=op, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return OP_SUCCESS, f'{op} task cache directory: {pro_dir}'

    def evaluate(self, op, user, uuid, params):
        logger.info("call evaluate(%s, %s, %s)", op, user, uuid)
        if op == 'evaluate.stop':
            Thread(target=lambda: self._run(task=op, user=user, uuid=uuid),
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
        Thread(target=lambda: self._run(task=op, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return OP_SUCCESS, f'{op} task cache directory: {pro_dir}'

    def predict(self, op, user, uuid, params):
        logger.info("call predict(%s, %s, %s)", op, user, uuid)
        if op == 'predict.stop':
            Thread(target=lambda: self._run(task=op, user=user, uuid=uuid),
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
        Thread(target=lambda: self._run(task=op, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return OP_SUCCESS, f'{op} task cache directory: {pro_dir}'

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
            image=image, libs='hzcsnlp', debug=LEVEL==logging.DEBUG))
        app.bind('tcp://%s:%d' % (host, args.port))
        app.run()
    finally:
        app_quit = True
        thread.join()
