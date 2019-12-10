#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-20 18:22:48

import os, sys, time
import argparse
import logging, json
import zerorpc
import requests
import consul
import docker
import GPUtil
import psutil
from subprocess import Popen, PIPE
from threading import Thread

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

platform_service_name = 'k12platform'

if os.environ.get("K12PLATFORM_DEBUG"):
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

def _delay_do_consul(host, port):
    time.sleep(3)
    while not app_quit:
        try:
            client = consul.Consul(host=consul_addr, port=consul_port)
            client.agent.service.register(
                    name='{}-{}'.format(app_host_name, platform_service_name),
                    address=host, port=port, tags=('AI', 'PLATFORM'),
                    check=consul.Check.tcp(host, port,
                        interval='10s', timeout='5s', deregister='10s'))
            break
        except Exception as err:
            logger.info("consul agent service register err", err)
            time.sleep(3)

OP_SUCCESS = 0
OP_FAILURE = -1

def _get_container_mem_pct(stats):
    usage = stats['memory_stats']['usage']
    limit = stats['memory_stats']['limit']
    return round(usage * 100 / limit, 2)

def _get_container_cpu_pct(stats):
    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
        stats['precpu_stats']['cpu_usage']['total_usage']
    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
        stats['precpu_stats']['system_cpu_usage']
    online_cpus = stats['cpu_stats']['online_cpus']
    if cpu_delta > 0 and system_delta > 0:
        return round((cpu_delta / system_delta) * online_cpus * 100, 2)
    return 0.0

def _get_cpu_infos():
    info = {}
    info['cpu_percent'] = psutil.cpu_percent()
    info['cpu_percent_list'] = psutil.cpu_percent(percpu=True)
    info['cpu_memory_total'] = psutil.virtual_memory().total
    info['cpu_memory_usage'] = psutil.virtual_memory().used
    info['cpu_memory_percent'] = psutil.virtual_memory().percent
    return info

def _get_gpu_infos():
    infos = []
    for g in GPUtil.getGPUs():
        info = {}
        info['name'] = g.name
        info['gpu_percent'] = round(g.load * 100, 2)
        info['gpu_memory_total'] = g.memoryTotal
        info['gpu_memory_usage'] = g.memoryUsed
        info['gpu_memory_percent'] = round(g.memoryUtil * 100, 2)
        infos.append(info)
    return infos

def _get_disk_infos():
    infos = []
    for d in psutil.disk_partitions():
        info = {}
        usage = psutil.disk_usage(d.mountpoint)
        info['path'] = d.mountpoint
        info['disk_total'] = usage.total
        info['disk_usage'] = usage.used
        info['disk_percent'] = usage.percent
        infos.append(info)
    return infos

def _get_process_infos():
    infos = []
    try:
        process = Popen(['nvidia-smi', "pmon", "--count", "1", "--select", "mu"], stdout=PIPE)
        stdout, stderr = process.communicate() 
        output = stdout.decode('UTF-8').split('\n')
        for i in range(2, len(output)-1):
            info = {}
            result = output[i].split()
            if len(result) > 5:
                info['pid'] = result[1]
                info['gpu_percent'] = float(result[4])
                info['gpu_memory_usage'] = int(result[3])
                info['gpu_memory_percent'] = float(result[5])
            infos.append(info)
    except Exception as err:
        logger.error(str(err))
    return infos

def _get_container_infos(client):
    infos = []
    try:
        cons = client.containers.list(filters={'label': 'k12ai.service.name'})
        if len(cons) == 0:
            return infos
        gpu_process_infos = _get_process_infos()
        for c in cons:
            info = {}
            stats = c.stats(stream=False)
            info['id'] = stats['id'][0:12]
            info['op'] = c.labels.get('k12ai.service.op', '')
            info['user'] = c.labels.get('k12ai.service.user', '')
            info['service_uuid'] = c.labels.get('k12ai.service.uuid', '')
            info['cpu_percent'] = _get_container_cpu_pct(stats)
            info['cpu_memory_total'] = stats['memory_stats']['limit']
            info['cpu_memory_usage'] = stats['memory_stats']['usage']
            info['cpu_memory_percent'] = _get_container_mem_pct(stats)
            info['gpu_percent'] = 0.00
            info['gpu_memory_usage'] = 0
            info['gpu_memory_percent'] = 0.00
            
            pids = sum(c.top(ps_args='eo pid')['Processes'], [])
            for ginfo in gpu_process_infos:
                if ginfo['pid'] in pids:
                    info['gpu_percent'] += ginfo['gpu_percent']
                    info['gpu_memory_usage'] += ginfo['gpu_memory_usage']
                    info['gpu_memory_percent'] += ginfo['gpu_memory_percent']
            infos.append(info)
    except Exception as err:
        logger.error(str(err))
    return infos

class PlatformServiceRPC(object):

    def __init__(self, host, port, k12ai='k12ai', debug=False):
        self._debug = debug
        self._host = host
        self._port = port
        self._k12ai = k12ai
        self._docker = docker.from_env()

    def send_message(self, op, user, uuid, msgtype, message):
        client = consul.Consul(consul_addr, port=consul_port)
        service = client.agent.services().get(self._k12ai)
        if not service:
            logger.error("Not found %s service!" % self._k12ai)
            return

        data = {
                'version': '0.1.0',
                'type': msgtype,
                'tag': 'platform',
                'user': user,
                'service_uuid': uuid,
                'op': op
                }
        now_time = time.time()
        data['timestamp'] = round(now_time * 1000)
        data['datetime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_time))
        data[msgtype] = message

        # service
        api = 'http://{}:{}/k12ai/private/message'.format(service['Address'], service['Port'])
        requests.post(api, json=data)
        if self._debug:
            key = 'platform/%s/%s/%s/%s' % (user, uuid, op, msgtype)
            client.kv.put(key, json.dumps(data, indent=4))

    def _run_stats(self, op, user, uuid, params):
        message = {}
        if op == 'query':
            if not params:
                message = _get_cpu_infos()
                message['gpus'] = _get_gpu_infos()
                message['disks'] = _get_disk_infos()
                message['containers'] = _get_container_infos(self._docker)
            elif isinstance(params, dict):
                if params.get('cpus', True):
                    message = _get_cpu_infos()
                if params.get('gpus', True):
                    message['gpus'] = _get_gpu_infos()
                if params.get('disks', True):
                    message['disks'] = _get_disk_infos()
                if params.get('containers', True):
                    message['containers'] = _get_container_infos(self._docker)
            self.send_message(op, user, uuid, 'result', _err_msg(message=message))
        else:
            message['todo'] = 'TODO'
            self.send_message(op, user, uuid, 'error', message)
            return OP_FAILURE, message
        return OP_SUCCESS, message

    def stats(self, op, user, uuid, params, isasync):
        logger.info("call stats, isasync: {}".format(isasync))
        if not isasync:
            return self._run_stats(op, user, uuid, params)
        Thread(target=lambda: self._run_stats(op, user, uuid, params), daemon=True).start()
        return OP_SUCCESS, None

    def _run_stop_container(self, op, user, uuid, params):
        message = None
        code = OP_SUCCESS
        try:
            cid = params.get('id', None)
            con = self._docker.containers.get(cid)
            if con.status == 'running':
                con.kill()
            message = f'container id {cid} stop success'
        except docker.errors.NotFound:
            code = OP_FAILURE
            message = f'container id {cid} is not found'

        self.send_message(op, user, uuid, 'result', message)
        return code, message

    def control(self, op, user, uuid, params, isasync):
        logger.info("call control op:{}, isasync:{}".format(op, isasync))
        if op == 'container.stop':
            if not isasync:
                return self._run_stop_container(op, user, uuid, params)
            Thread(target=lambda: self._run_stop_container(op, user, uuid, params), daemon=True).start()
        else:
            return OP_FAILURE, '{op} is not implemented yet'

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
            default=8129,
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

    args = parser.parse_args()

    host = args.host if args.host else app_host_ip

    consul_addr = args.consul_addr if args.consul_addr else app_host_ip
    consul_port = args.consul_port

    thread = Thread(target=_delay_do_consul, args=(host, args.port))
    thread.start()

    logger.info('start zerorpc server on %s:%d', host, args.port)

    try:
        app = zerorpc.Server(PlatformServiceRPC(
            host=host, port=args.port,
            k12ai='{}-k12ai'.format(app_host_name),
            debug=LEVEL==logging.DEBUG)) # noqa
        app.bind('tcp://%s:%d' % (host, args.port))
        app.run()
    finally:
        app_quit = True
        thread.join()
