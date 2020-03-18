#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_platform.py
# @brief
# @author QRS
# @blog qrsforever.github.io
# @version 1.0
# @date 2020-01-21 13:57

import docker
import GPUtil
import psutil
from subprocess import Popen, PIPE
from threading import Thread

from k12ai.k12ai_errmsg import k12ai_error_message
from k12ai.k12ai_utils import k12ai_utils_lanip
from k12ai.k12ai_consul import k12ai_consul_message

g_cpu_count = -1
g_gpu_count = -1
g_docker = None


def MB(d, n=3):
    return round(d / 1024**2, n)


def _get_docker():
    global g_docker
    if not g_docker:
        g_docker = docker.from_env()
    return g_docker


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
    info['ip'] = k12ai_utils_lanip()
    info['cpu_percent'] = psutil.cpu_percent()
    info['cpu_percent_list'] = psutil.cpu_percent(percpu=True)
    info['cpu_memory_total_MB'] = MB(psutil.virtual_memory().total)
    info['cpu_memory_usage_MB'] = MB(psutil.virtual_memory().used)
    info['cpu_memory_free_MB'] = MB(psutil.virtual_memory().available)
    info['cpu_memory_percent'] = psutil.virtual_memory().percent
    return info


def _get_gpu_infos():
    infos = []
    gpus = GPUtil.getGPUs()
    for g in gpus:
        info = {}
        info['name'] = g.name
        info['gpu_percent'] = round(g.load * 100, 2)
        info['gpu_memory_total_MB'] = g.memoryTotal
        info['gpu_memory_usage_MB'] = g.memoryUsed
        info['gpu_memory_free_MB'] = g.memoryFree
        info['gpu_memory_percent'] = round(g.memoryUtil * 100, 2)
        infos.append(info)
    if len(gpus) == 1:
        return infos[0]
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
        for i in range(2, len(output) - 1):
            info = {}
            result = output[i].split()
            if len(result) > 5:
                info['idx'] = result[0]
                info['pid'] = result[1]
                info['gpu_percent'] = float(result[4])
                info['gpu_memory_usage_MB'] = int(result[3])
                info['gpu_memory_percent'] = float(result[5])
            infos.append(info)
    except Exception as err:
        print('error:{}'.format(err))
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
            info['cpu_memory_total_MB'] = MB(stats['memory_stats']['limit'])
            info['cpu_memory_usage_MB'] = MB(stats['memory_stats']['usage'])
            info['cpu_memory_percent'] = _get_container_mem_pct(stats)
            info['gpu_percent'] = 0.00
            info['gpu_memory_usage_MB'] = 0
            info['gpu_memory_percent'] = 0.00

            pids = sum(c.top(ps_args='eo pid')['Processes'], [])
            for ginfo in gpu_process_infos:
                if ginfo['pid'] in pids:
                    info['gpu_percent'] += ginfo['gpu_percent']
                    info['gpu_memory_usage_MB'] += ginfo['gpu_memory_usage_MB']
                    info['gpu_memory_percent'] += ginfo['gpu_memory_percent']
            infos.append(info)
    except Exception as err:
        print('error:{}'.format(err))
    return infos


def _get_service_infos(client, user, uuid):
    infos = []
    try:
        cons = []
        if uuid == '0' or uuid == '*' or uuid == 'all':
            cons = client.containers.list(filters={'label': 'k12ai.service.user=%s'%user})
        else:
            cons = client.containers.list(filters={'label': ['k12ai.service.user=%s'%user, 'k12ai.service.uuid=%s'%uuid]})
        if len(cons) == 0:
            return infos
        for c in cons:
            info = {}
            stats = c.stats(stream=False)
            info['id'] = stats['id'][0:12]
            info['op'] = c.labels.get('k12ai.service.op', '')
            info['service_uuid'] = c.labels.get('k12ai.service.uuid', '')
            info['service_pid'] = c.attrs.get('State', {}).get('Pid', -1)
            info['service_starttime'] = c.attrs.get('State', {}).get('StartedAt', '')
            infos.append(info)
    except Exception as err:
        print('error:{}'.format(err))
    return infos


def _query_stats(docker, op, user, uuid, params):
    message = {}
    if not params:
        message = _get_cpu_infos()
        message['gpus'] = _get_gpu_infos()
        message['disks'] = _get_disk_infos()
        message['containers'] = _get_container_infos(docker)
    elif isinstance(params, dict):
        if params.get('cpus', False):
            message.update(_get_cpu_infos())
        if params.get('gpus', False):
            message['gpus'] = _get_gpu_infos()
        if params.get('disks', False):
            message['disks'] = _get_disk_infos()
        if params.get('containers', False):
            message['containers'] = _get_container_infos(docker)
        if params.get('services', False):
            message['services'] = _get_service_infos(docker, user, uuid)
        k12ai_consul_message('k12ai', uuid, op, user, uuid, 'resource', k12ai_error_message(content=message), clear=True)
    return 100000, message


def _stop_container(op, user, uuid, params):
    try:
        cid = params.get('id', None)
        con = _get_docker().containers.get(cid)
        if con.status == 'running':
            con.kill()
    except docker.errors.NotFound:
        return 100205, f'container:{cid}'
    return 100000, None


def k12ai_platform_stats(op, user, uuid, params, isasync):
    if op not in ('query'):
        return 100902, None

    if op == 'query':
        docker = _get_docker()
        if isasync:
            Thread(target=lambda: _query_stats(docker, op, user, uuid, params), daemon=True).start()
            return 100000, None
        return _query_stats(docker, op, user, uuid, params)


def k12ai_platform_control(op, user, uuid, params, isasync):
    if op not in ('container.stop'):
        return 100902, None

    if op == 'container.stop':
        return _stop_container(op, user, uuid, params)


def k12ai_platform_cpu_count():
    global g_cpu_count
    if g_cpu_count < 0:
        g_cpu_count = psutil.cpu_count()
    return g_cpu_count


def k12ai_platform_gpu_count():
    global g_gpu_count
    if g_gpu_count < 0:
        g_gpu_count = len(GPUtil.getGPUs())
    return g_gpu_count


def k12ai_platform_memory_free(unit='M'):
    if unit not in ('M'):
        raise AssertionError
    sys_gpu_mfree = []
    for i, g in enumerate(GPUtil.getGPUs(), 0):
        sys_gpu_mfree.append(GPUtil.getGPUs()[i].memoryFree)
    result = {
        'sys_cpu_memory_free_MB': MB(psutil.virtual_memory().available),
        'sys_gpu_memory_free_MB': sys_gpu_mfree,
    }
    return result


def k12ai_platform_memory_stat(container):
    infos = {}
    if not container:
        return infos
    memused = 0.0
    memfree = 0.0
    gpu_process_infos = _get_process_infos()
    pids = sum(container.top(ps_args='eo pid')['Processes'], [])
    for ginfo in gpu_process_infos:
        if ginfo['pid'] in pids:
            memused += ginfo['gpu_memory_usage_MB']
    gpus = GPUtil.getGPUs()
    for i, g in enumerate(gpus, 0):
        memfree += g.memoryFree
    infos['app_gpu_memory_usage_MB'] = memused
    infos['sys_cpu_memory_free_MB'] = MB(psutil.virtual_memory().available)
    infos['sys_gpu_memory_free_MB'] = memfree
    return infos
