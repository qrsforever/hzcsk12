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
            logger.info('debug mode')
            self._workdir = workdir
            self._projdir = os.path.abspath( # noqa: E126
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
        data[msgtype] = message

        # service
        api = 'http://{}:{}/k12ai/private/message'.format(service['Address'], service['Port'])
        requests.post(api, json=data)
        if self._debug:
            key = 'framework/%s/%s/%s/%s' % (user, uuid, op, msgtype)
            if msgtype != 'status':
                key = '%s/%s' % (key, data['datetime'][:-2])
            client.kv.put(key, json.dumps(data, indent=4))

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

    def _prepare_environ(self, user, uuid, params):
        try:
            if not params or not isinstance(params, dict):
                return OP_FAILURE, _err_msg(100203, 'parameter must be dict type')

            task = params['task']
            if task not in ['cls', 'det', 'seg', 'pose', 'gan']:
                return OP_FAILURE, _err_msg(100203, f'task[{{task}}] is not support yet')

            pro_dir = os.path.join(users_cache_dir, user, uuid)
            if not os.path.exists(pro_dir):
                os.makedirs(pro_dir)

            dataset_path = params['data']['data_dir']

            if not os.path.exists(dataset_path):
                return OP_FAILURE, _err_msg(100203, f'dataset [{dataset_path}] is not exist!')

            dataset_name = dataset_path.split('/')[-1]

            params['cache_dir'] = os.path.join(users_cache_dir, user, uuid)
            params['network']['checkpoints_dir'] = 'ckpts/' + dataset_name
            config_path = os.path.join(pro_dir, 'config.json')
            with open(config_path, 'w') as fout:
                fout.write(json.dumps(params))

            ckpts_root = params['cache_dir']
            ckpts_dirx = params['network']['checkpoints_dir']
            ckpts_name = params['network']['model_name']
            resume_path = os.path.join(ckpts_root, ckpts_dirx, ckpts_name + '_latest.pth')

            if task == 'cls':
                test_dir = os.path.join(dataset_path, 'imgs', 'test')
            else:
                test_dir = dataset_path

            out_dir = os.path.join(ckpts_root, 'out')

            return OP_SUCCESS, { # noqa: E126
                    'config_path': config_path,
                    'resume_path': resume_path,
                    'dataset_path': dataset_path,
                    'ckpts_root': ckpts_root,
                    'ckpts_name': ckpts_name,
                    'test_dir': test_dir,
                    'out_dir': out_dir
                    }
        except Exception:
            return OP_FAILURE, _err_msg(100203, 'prepare environ occur exception', exc=True)

    def _run(self, op, user, uuid, command=None):
        logger.info(command)
        message = None
        container_name, con = self._get_container(op, user, uuid)

        if not command: # stop
            try:
                if con:
                    if con.status == 'running':
                        con.kill()
                        message = _err_msg(100300, f'container name:{container_name}')
                        xop = op.replace('stop', 'start')
                        self.send_message(xop, user, uuid, "status", {'value':'exit', 'way': 'manual'})
                else:
                    message = _err_msg(100301, f'container name:{container_name}')
            except Exception:
                message = _err_msg(100303, f'container name:{container_name}', exc=True)
        else: # start
            rm_flag = True
            labels = { # noqa
                    'k12ai.service.name': service_name,
                    'k12ai.service.op': op,
                    'k12ai.service.user': user,
                    'k12ai.service.uuid': uuid
                    }
            volumes = { # noqa
                    '/data': {'bind':'/data', 'mode':'rw'}
                    }
            if self._debug:
                rm_flag = False
                volumes['%s/app'%self._projdir] = {'bind':'%s/app'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/data'%self._projdir] = {'bind':'%s/torchcv/data'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/metric'%self._projdir] = {'bind':'%s/torchcv/metric'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/model'%self._projdir] = {'bind':'%s/torchcv/model'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/runner'%self._projdir] = {'bind':'%s/torchcv/runner'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/tools'%self._projdir] = {'bind':'%s/torchcv/tools'%self._workdir, 'mode':'rw'}
                volumes['%s/torchcv/main.py'%self._projdir] = {'bind':'%s/torchcv/main.py'%self._workdir, 'mode':'rw'}
            environs = {
                    'K12CV_RPC_HOST': '%s' % self._host,
                    'K12CV_RPC_PORT': '%s' % self._port,
                    'K12CV_OP': '%s' % op,
                    'K12CV_USER': '%s' % user,
                    'K12CV_UUID': '%s' % uuid
                    } # noqa
            kwargs = {
                    'name': container_name,
                    'auto_remove': rm_flag,
                    'detach': True,
                    'runtime': 'nvidia',
                    'labels': labels,
                    'volumes': volumes,
                    'environment': environs
                    } # noqa

            if con and con.status == 'running':
                message = _err_msg(100304, 'container name: {}'.format(con.short_id))
            else:
                self.send_message(op, user, uuid, "status", {'value':'starting'})
                try:
                    if con:
                        con.remove()
                    con = self._docker.containers.run(self._image,
                            command, **kwargs)
                    return
                except Exception:
                    message = _err_msg(100302, 'container image:{}'.format(self._image), exc=True)
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

        code, result = self._prepare_environ(user, uuid, params)
        if code < 0:
            return OP_FAILURE, json.dumps(result)

        resume_flag = True
        if not os.path.exists(result['resume_path']):
            resume_flag = False

        command = 'python {} {} {} {} {} {} --dist y --gather y --phase train'.format(
                '-m torch.distributed.launch --nproc_per_node=1',
                '/hzcsk12/cv/torchcv/main.py',
                '--config_file %s' % result['config_path'],
                '--checkpoints_root %s' % result['ckpts_root'],
                '--checkpoints_name %s' % result['ckpts_name'],
                '{}'.format('--resume %s' % result['resume_path'] if resume_flag else ' ')
                ) # noqa

        Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return OP_SUCCESS, result

    def evaluate(self, op, user, uuid, params):
        logger.info("call evaluate(%s, %s, %s)", op, user, uuid)
        if op == 'evaluate.stop':
            Thread(target=lambda: self._run(op=op, user=user, uuid=uuid),
                    daemon=True).start()
            return OP_SUCCESS, None

        code, result = self._prepare_environ(user, uuid, params)
        if code < 0:
            return OP_FAILURE, json.dumps(result)

        if not os.path.exists(result['resume_path']):
            return OP_FAILURE, json.dumps(_err_msg(100203, 'resume path[%s] is not found!' % result['resume_path']))

        command = 'python {} {} {} {} {} {} {} {} --dist y --gather y --phase test'.format(
                '-m torch.distributed.launch --nproc_per_node=1',
                '/hzcsk12/cv/torchcv/main.py',
                '--config_file %s' % result['config_path'],
                '--checkpoints_root %s' % result['ckpts_root'],
                '--checkpoints_name %s' % result['ckpts_name'],
                '--resume %s' % result['resume_path'],
                '--test_dir %s' % result['test_dir'],
                '--out_dir %s' % result['out_dir']
                ) # noqa

        Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
                daemon=True).start()
        return OP_SUCCESS, result

    def predict(self, op, user, uuid, params):
        logger.info("call predict(%s, %s, %s)", op, user, uuid)
        if op == 'predict.stop':
            Thread(target=lambda: self._run(op=op, user=user, uuid=uuid),
                    daemon=True).start()
            return OP_SUCCESS, None

        if not params or not isinstance(params, dict):
            return OP_FAILURE, 'params is none or not dict type'

        # Thread(target=lambda: self._run(op=op, user=user, uuid=uuid, command=command),
        #         daemon=True).start()
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
            image=image, debug=LEVEL==logging.DEBUG)) # noqa
        app.bind('tcp://%s:%d' % (host, args.port))
        app.run()
    finally:
        app_quit = True
        thread.join()
