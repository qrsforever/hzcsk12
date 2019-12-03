#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12cv_service.py
# @brief
# @author QRS
# @blog qrsforever.github.io
# @version 1.0
# @date 2019-11-27 17:08:18

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

service_name = 'k12cv'

if os.environ.get("K12CV_DEBUG"):
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

OP_SUCCESS = 0
OP_FAILURE = -1

class CVServiceRPC(object):

    def __init__(self,
            host, port,
            k12ai='k12ai',
            image='hzcsai_com/k12cv',
            workdir='/hzcsk12/cv', debug=False):
        self._debug = debug
        self._host = host
        self._port = port
        self._k12ai = k12ai
        self._image = image
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
                code = 100305
            elif cls == 'MemoryError':
                code = 100901
            else:
                code = 100399
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

    def _get_container(self, task, user, uuid):
        container_name = '%s-%s-%s' % (task.split('.')[0], user, uuid)
        try:
            con = self._docker.containers.get(container_name)
            return container_name, con
        except docker.errors.NotFound:
            return container_name, None

    def _run(self, task, user, uuid, command=None):
        message = {}
        stopcmd = command == None
        container_name, con = self._get_container(task, user, uuid)

        if stopcmd: # stop
            try:
                if con:
                    if con.status == 'running':
                        # con.stop()
                        con.kill()
                        message = _err_msg(100300, f'container name:{container_name}')
                else:
                    message = _err_msg(100301, f'container name:{container_name}')
            except Exception as err:
                message = _err_msg(100303, f'container name:{container_name}', exc=True)
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
                volumes['%s/torchcv/datasets'%self._projdir] = {'bind':'%s/torchcv/datasets'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/metric'%self._projdir] = {'bind':'%s/torchcv/metric'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/model'%self._projdir] = {'bind':'%s/torchcv/model'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/runner'%self._projdir] = {'bind':'%s/torchcv/runner'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/tools'%self._projdir] = {'bind':'%s/torchcv/tools'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/main.py'%self._projdir] = {'bind':'%s/torchcv/main.py'%self._workdir, 'mode':'rw'}
            environs = {
                    'K12CV_RPC_HOST': '%s' % self._host,
                    'K12CV_RPC_PORT': '%s' % self._port,
                    'K12CV_TASK': '%s' % task,
                    'K12CV_USER': '%s' % user,
                    'K12CV_UUID': '%s' % uuid
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
                    return
                else:
                    if con:
                        message = _err_msg(100304, 'container name: {}'.format(con.short_id))
            except Exception as err:
                message = _err_msg(100302, 'container image:{}'.format(self._image), exc=True)

        self.send_message(task, user, uuid, "error", message)

    def train(self, op, user, uuid, params):
        logger.info("call train(%s, %s, %s)", op, user, uuid)
        if op == 'train.stop':
            Thread(target=lambda: self._run(task=op, user=user, uuid=uuid),
                    daemon=True).start()
            return OP_SUCCESS, None

        container_name, con = self._get_container(op, user, uuid)
        if con and con.status == 'running':
            return OP_FAILURE, f'[{container_name}] same task is running!'

        if not params or not isinstance(params, dict):
            return OP_FAILURE, 'parameter is none or not dict type'

        pro_dir = os.path.join(users_cache_dir, user, uuid)
        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)

        dataset_name = params['dataset']
        dataset_path = params['data']['data_dir']

        if not os.path.exists(dataset_path):
            return OP_FAILURE, f'path[{dataset_path}] is not exist!'

        params['network']['checkpoints_dir'] = 'ckpts/' + dataset_name
        training_conf = os.path.join(pro_dir, 'config.json')
        with open(training_conf, 'w') as fout:
            fout.write(json.dumps(params))

        ckpt_root = '%s/%s/%s'%(users_cache_dir, user, uuid)
        ckpt_dirx = params['network']['checkpoints_dir']
        ckpt_name = params['network']['model_name']
        resume_path = os.path.join(ckpt_root, ckpt_dirx, ckpt_name + '_latest.pth')
        resume_flag = True
        if not os.path.exists(resume_path):
            resume_flag = False

        command = 'python {} {} {} {} {} {} --dist y --gather y --phase train'.format(
                '-m torch.distributed.launch --nproc_per_node=1',
                '/hzcsk12/cv/torchcv/main.py',
                '--config_file %s' % training_conf,
                '--checkpoints_root %s' % ckpt_root,
                '--checkpoints_name %s' % ckpt_name,
                '{}'.format('--resume %s' % resume_path if resume_flag else ' ')
                )

        logger.info(command)

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

        training_conf = os.path.join(users_cache_dir, user, uuid, 'config.json')
        with open(training_conf, 'w') as fout:
            fout.write(json.dumps(params))

        dataset_name = params['dataset']
        dataset_path = params['data']['data_dir']

        if not os.path.exists(dataset_path):
            return OP_FAILURE, f'path[{dataset_path}] is not exist!'

        ckpt_root = '%s/%s/%s'%(users_cache_dir, user, uuid)
        ckpt_dirx = params['network']['checkpoints_dir']
        ckpt_name = params['network']['model_name']
        resume_path = os.path.join(ckpt_root, ckpt_dirx, ckpt_name + '_latest.pth')
        resume_flag = True
        if not os.path.exists(resume_path):
            return OP_FAILURE, f'path[{resume_path}] is not exist!'

        test_dir = os.path.join(dataset_path, 'imgs', 'test')
        if not os.path.exists(test_dir):
            return OP_FAILURE, f'path[{test_path}] is not exist!'

        command = 'python {} {} {} {} {} {} {} --gather n --phase test'.format(
                '-m torch.distributed.launch --nproc_per_node=1',
                '/hzcsk12/cv/torchcv/main.py',
                '--config_file %s' % training_conf,
                '--checkpoints_root %s' % ckpt_root,
                '--checkpoints_name %s' % ckpt_name,
                '--resume %s' % resume_path,
                '--test_dir %s' % test_dir
                )

        logger.info(command)

        Thread(target=lambda: self._run(task=op, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return OP_SUCCESS, None

    def predict(self, op, user, uuid, params):
        logger.info("call predict(%s, %s, %s)", op, user, uuid)
        if op == 'predict.stop':
            Thread(target=lambda: self._run(task=op, user=user, uuid=uuid),
                    daemon=True).start()
            return OP_SUCCESS, None

        if not params or not isinstance(params, dict):
            return OP_FAILURE, 'params is none or not dict type'

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
            default=None,
            type=str,
            dest='image',
            help="image to run container")
    args = parser.parse_args()

    image = args.image if args.image else 'hzcsai_com/k12cv'

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
            image=image, debug=LEVEL==logging.DEBUG))
        app.bind('tcp://%s:%d' % (host, args.port))
        app.run()
    finally:
        app_quit = True
        thread.join()
