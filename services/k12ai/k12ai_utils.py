#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file k12ai_utils.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-12-07 09:45:23

import time
import socket
import os
import errno
import json
from minio import Minio
import tarfile

_LANIP = None
_NETIP = None


def k12ai_timeit(handler):
    def decorator(func):
        def timed(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            if handler:
                handler('"{}" took {:.3f} ms to execute'.format(func.__name__, (te - ts) * 1000))
            return result
        return timed
    return decorator


def k12ai_utils_topdir():
    return os.path.abspath(
                os.path.dirname(os.path.abspath(__file__)) + '/../..')


def k12ai_utils_hostname():
    val = os.environ.get('HOST_NAME', None)
    if val:
        return val
    return socket.gethostname()


def k12ai_utils_lanip():
    global _LANIP
    if _LANIP:
        return _LANIP
    val = os.environ.get('HOST_LANIP', None)
    if not val:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8',80))
            val = s.getsockname()[0]
        finally:
            s.close()
    _LANIP = val
    return _LANIP


def k12ai_utils_netip():
    global _NETIP
    if _NETIP:
        return _NETIP
    val = os.environ.get('HOST_NETIP', None)
    if not val:
        result = os.popen('curl -s http://txt.go.sohu.com/ip/soip| grep -P -o -i "(\d+\.\d+.\d+.\d+)"', 'r') # noqa
        if result:
            val = result.read().strip('\n')
    _NETIP = val
    return _NETIP


def k12ai_utils_diff(conf1, conf2):
    if isinstance(conf1, dict):
        param1 = conf1
    else:
        if not os.path.exists(conf1):
            return True
        with open(conf1, 'r') as f1:
            param1 = json.loads(f1.read())

    if isinstance(conf2, dict):
        param2 = conf2
    else:
        if not os.path.exists(conf2):
            return True
        with open(conf2, 'r') as f2:
            param2 = json.loads(f2.read())

    diff = False
    for key in param1.keys() - param2.keys():
        print(f"Key '{key}' found in training configuration but not in the serialization "
                     f"directory we're recovering from.")
        diff = True

    for key in param1.keys() - param2.keys():
        print(f"Key '{key}' found in the serialization directory we're recovering from "
                     f"but not in the training config.")
        diff = True

    for key in param1.keys():
        if param1.get(key, None) != param2.get(key, None):
            print(f"Value for '{key}' in training configuration does not match that the value in "
                         f"the serialization directory we're recovering from: "
                         f"{param1[key]} != {param2[key]}")
            diff = True
    return diff


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def k12ai_oss_client(server_url=None, access_key=None, secret_key=None,
        region='gz', bucket='k12ai'):
    if server_url is None:
        server_url = os.environ.get('MINIO_SERVER_URL')
    if access_key is None:
        access_key = os.environ.get('MINIO_ACCESS_KEY')
    if secret_key is None:
        secret_key = os.environ.get('MINIO_SECRET_KEY')

    mc = Minio(
        endpoint=server_url,
        access_key=access_key,
        secret_key=secret_key,
        secure=True)

    if not mc.bucket_exists(bucket):
        mc.make_bucket(bucket, location=region)

    return mc


def k12ai_object_put(client, local_path,
        bucket_name='k12ai', prefix_map=None,
        content_type='application/octet-stream', metadata=None):

    result = []

    def _upload_file(local_file):
        if not os.path.isfile(local_file):
            return
        if prefix_map and isinstance(prefix_map, list):
            lprefix = prefix_map[0].rstrip(os.path.sep)
            rprefix = prefix_map[1].strip(os.path.sep)
            remote_file = local_file.replace(lprefix, rprefix, 1)
        else:
            remote_file = local_file.lstrip(os.path.sep)

        file_size = os.stat(local_file).st_size
        with open(local_file, 'rb') as file_data:
            btime = time.time()
            etag = client.put_object(bucket_name,
                    remote_file, file_data, file_size,
                    content_type=content_type, metadata=metadata)
            etime = time.time()
            result.append({'etag': etag,
                'file': remote_file,
                'size': file_size,
                'time': [btime, etime]})

    if os.path.isdir(local_path):
        for root, directories, files in os.walk(local_path):
            for filename in files:
                _upload_file(os.path.join(root, filename))
    else:
        _upload_file(local_path)

    return result


def k12ai_object_get(client, remote_path,
        bucket_name='k12ai', prefix_map=None):

    remote_path = remote_path.lstrip(os.path.sep)

    result = []
    for obj in client.list_objects(bucket_name, prefix=remote_path, recursive=True):
        if prefix_map:
            local_file = obj.object_name.replace(prefix_map[0], prefix_map[1], 1)
        else:
            local_file = '/' + obj.object_name
        dfile = os.path.dirname(local_file)
        if dfile:
            mkdir_p(dfile)
        btime = time.time()
        data = client.get_object(bucket_name, obj.object_name)
        with open(local_file, 'wb') as file_data: 
            for d in data.stream():
                file_data.write(d)
        etime = time.time()
        result.append({'etag': obj.etag,
            'file': local_file,
            'size': obj.size,
            'time': [btime, etime]})
    return result


def k12ai_object_remove(client, remote_path, bucket_name='k12ai'):
    for obj in client.list_objects(bucket_name, prefix=remote_path, recursive=True):
        client.remove_object(obj.bucket_name, obj.object_name)
