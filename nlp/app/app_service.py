#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-15 16:38:16

import os, sys, time
import argparse
import signal
import logging, json
import socket
import zerorpc
import requests
import consul
import subprocess
import docker
from threading import Thread

if os.environ.get("K12NLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

logger = logging.getLogger(__name__)

app_quit = False

def _get_host_name():
    return socket.gethostname()

def _get_host_ip():
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

userdata_dir = '/data/users'

def _delay_do_consul(host, port):
    time.sleep(3)
    while not app_quit:
        try:
            client = consul.Consul(host=consul_addr, port=consul_port)
            client.agent.service.register(
                    name='{}-k12nlp'.format(socket.gethostname()),
                    address=host, port=port, tags=('AI', 'ML'),
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
        self._k12ai = '{}-k12ai'.format(app_host_name)
        self._image = image
        self._libs = libs
        self._docker = docker.from_env()
        self._workdir = workdir
        self._projdir = os.path.abspath(
                os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "..")
        logger.info('workdir:%s, projdir:%s', self._workdir, self._projdir)

    def send_message(self, task, user, uuid, message):
        client = consul.Consul(consul_addr, port=consul_port)
        service = client.agent.services().get(self._k12ai)
        if not service:
            logger.error("Not found %s service!" % self._k12ai)
            return
        # service
        api = 'http://{}:{}/k12ai/framework/message'.format(service['Address'], service['Port'])
        requests.post(api, json=message)
        client.kv.put('%s/%s/%s'%(user, uuid, task), json.dumps(message))

    def _run(self, task, user, uuid, command=None):
        message = {
                'user': user,
                'service_uuid': uuid,
                'timestamp': round(time.time() * 1000)
                }
        container_id = '%s-%s-%s' % (task, user, uuid)
        if command == None:
            message['op'] = '%s.stop' % task
            try:
                con = self._docker.containers.get(container_id)
                con.stop()
                message['result'] = {'code': 0, 'id': con.id}
            except Exception as err:
                logger.error(err)
                message['result'] = {'code': -1, 'err': str(err)}
        else:
            message['op'] = '%s.start' % task
            volumes = {
                    '/data': {'bind':'/data', 'mode':'rw'}
                    }
            rm_flag = True
            if self._debug:
                # rm_flag = False
                volumes['%s/app'%self._projdir] = {'bind':'%s/app'%self._workdir, 'mode':'rw'}
                volumes['%s/allennlp'%self._projdir] = {'bind':'%s/allennlp'%self._workdir, 'mode':'rw'}
            logger.info(volumes)
            environs = {
                    'K12NLP_RPC_HOST': '%s' % self._host,
                    'K12NLP_RPC_PORT': '%s' % self._port,
                    'K12NLP_TASK': '%s' % task,
                    'K12NLP_USER': '%s' % user,
                    'K12NLP_UUID': '%s' % uuid
                    }
            kwargs = {
                    'name': container_id,
                    'auto_remove': rm_flag,
                    'detach': True,
                    'network_mode': 'host',
                    'runtime': 'nvidia',
                    'shm_size': '2g',
                    'volumes': volumes,
                    'environment': environs
                    }

            try:
                try:
                    con = self._docker.containers.get(container_id)
                    message['result'] = {'code': 1, 'err': 'Container for the task already running!'}
                except: 
                    con = self._docker.containers.run(self._image,
                            command, **kwargs)
                    message['result'] = {'code': 0, 'id': con.id}
            except Exception as err:
                logger.error(err)
                message['result'] = {'code': -1, 'err': str(err)}
        self.send_message(task, user, uuid, message)

    def train(self, op, user, uuid, params):
        logger.info("call train(%s, %s, %s)", op, user, uuid)
        task = 'train'
        if op == 'train.stop':
            Thread(target=lambda: self._run(task=task, user=user, uuid=uuid),
                    daemon=True).start()
            return 0

        pro_dir = os.path.join(userdata_dir, user, uuid)
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
        Thread(target=lambda: self._run(task=task, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return 0

    def evaluate(self, *args, **kwargs):
        logger.info("call evaluate")

    def predict(self, *args, **kwargs):
        logger.info("call predict")

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

    image, debug = (args.image, False) if args.image else ('hzcsai_com/k12nlp-dev', True)

    host = args.host if args.host else app_host_ip

    consul_addr = args.consul_addr if args.consul_addr else app_host_ip
    consul_port = args.consul_port

    thread = Thread(target=_delay_do_consul, args=(host, args.port))
    thread.start()

    logger.info('start zerorpc server on %s:%d', host, args.port)

    try:
        app = zerorpc.Server(NLPServiceRPC(
            host=host, port=args.port, k12ai='k12ai',
            image=image, libs='hzcsnlp', debug=debug))
        app.bind('tcp://%s:%d' % (host, args.port))
        app.run()
    finally:
        app_quit = True
        thread.join()
