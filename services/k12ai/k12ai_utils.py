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
import hashlib

_LANIP = None
_NETIP = None


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


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

XOS_APPID = None
XOS_REGION = None
XOS_SERVER_URL = None

def k12ai_oss_client(server_url=None, access_key=None, secret_key=None,
        region=None, bucket_name='k12ai'):
    global XOS_APPID, XOS_REGION, XOS_SERVER_URL
    if server_url is None:
        server_url = os.environ.get('XOS_SERVER_URL')
    if access_key is None:
        access_key = os.environ.get('XOS_ACCESS_KEY')
    if secret_key is None:
        secret_key = os.environ.get('XOS_SECRET_KEY')
    if region is None:
        region = os.environ.get('XOS_REGION')

    XOS_REGION = region
    XOS_SERVER_URL = server_url 
    if "myqcloud.com" in server_url:
        from qcloud_cos import CosConfig
        from qcloud_cos import CosS3Client
        appid = os.environ.get('XOS_APPID')
        mc = CosS3Client(CosConfig(
            Region=region,
            SecretId=access_key,
            SecretKey=secret_key,
            Token=None, Scheme='https'))
        XOS_APPID = appid
    else:
        from minio import Minio
        mc = Minio(
            endpoint=server_url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False)

        if not mc.bucket_exists(bucket_name):
            mc.make_bucket(bucket_name, location=region)

    return mc


def k12ai_object_put(client, local_path,
        bucket_name=None, prefix_map=None,
        content_type='application/octet-stream', metadata=None):

    if bucket_name is None:
        bucket_name = 'k12ai'

    if XOS_APPID is not None:
        bucket_name = f'{bucket_name}-{XOS_APPID}'
        xos_domain = f'https://{bucket_name}.cos.{XOS_REGION}.{XOS_SERVER_URL}'
    else:
        # s3-internal.didiyunapi.com
        xos_domain = f'https://{bucket_name}.{XOS_SERVER_URL}'

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
            if XOS_APPID is not None:
                response = client.put_object(
                        Bucket=bucket_name,
                        Body=file_data,
                        Key=remote_file)
                etag = response['ETag'].strip('"')
            else:
                etag = client.put_object(bucket_name,
                        remote_file, file_data, file_size,
                        content_type=content_type, metadata=metadata).etag
            etime = time.time()
            result.append({
                'etag': etag, 'bucket': bucket_name,
                'url': f'{xos_domain}/{remote_file}',
                'object': remote_file,
                'size': file_size,
                'time': [btime, etime]})

    if os.path.isdir(local_path):
        for root, directories, files in os.walk(local_path):
            for filename in files:
                _upload_file(os.path.join(root, filename))
    else:
        _upload_file(local_path)

    return result


def k12ai_object_list(client, prefix, recurive=False, bucket_name='k12ai'):
    if bucket_name is None:
        bucket_name = 'k12ai'

    objects = []
    if XOS_APPID is not None:
        bucket_name = f'{bucket_name}-{XOS_APPID}'
        delimiter = '' if recurive else '/'
        marker = ""
        while True:
            response = client.list_objects(Bucket=bucket_name, Prefix=prefix, Marker=marker, MaxKeys=50, Delimiter=delimiter)
            if 'Contents' in response:
                for content in response['Contents']:
                    objects.append(DotDict({
                        'etag': content['ETag'], 'size': content['Size'],
                        'object_name': content['Key'], 'bucket_name': bucket_name
                    }))
            if 'CommonPrefixes' in response:
                for folder in response['CommonPrefixes']:
                    pass
            if response['IsTruncated'] == 'false':
                break
            marker = response["NextMarker"]
    else:
        return list(client.list_objects(bucket_name, prefix=prefix, recursive=recurive))
    return objects


def k12ai_object_get(client, remote_path, bucket_name=None, prefix_map=None):
    if bucket_name is None:
        bucket_name = 'k12ai'
    remote_path = remote_path.lstrip(os.path.sep)

    if prefix_map:
        if prefix_map[0][0] == '/':
            prefix_map[0] = prefix_map[0][1:]
        if prefix_map[1][0] == '/':
            prefix_map[1] = prefix_map[1][1:]

    result = []
    for obj in k12ai_object_list(client, remote_path, True, bucket_name):
        if prefix_map:
            local_file = '/' + obj.object_name.replace(prefix_map[0], prefix_map[1], 1)
        else:
            local_file = '/' + obj.object_name
        if local_file[-1] == '/':
            continue
        dfile = os.path.dirname(local_file)
        if dfile:
            mkdir_p(dfile)
        btime = time.time()
        if XOS_APPID is not None:
            response = client.get_object(
        	Bucket=obj.bucket_name,
        	Key=obj.object_name
            )
            response['Body'].get_stream_to_file(local_file)
        else:
            data = client.get_object(bucket_name, obj.object_name)
            with open(local_file, 'wb') as file_data:
                for d in data.stream():
                    file_data.write(d)
        etime = time.time()
        result.append({'etag': obj.etag,
            'bucket': obj.bucket_name,
            'object': obj.object_name,
            'size': obj.size,
            'time': [btime, etime]})
    return result


def k12ai_object_remove(client, remote_path, bucket_name=None):
    if bucket_name is None:
        bucket_name = 'k12ai'
    result = []
    if remote_path[0] == '/':
        remote_path = remote_path[1:]
    for obj in k12ai_object_list(client, remote_path, True, bucket_name):
        client.remove_object(obj.bucket_name, obj.object_name)
        result.append({'etag': obj.etag,
            'bucket': obj.bucket_name,
            'object': obj.object_name,
            'size': obj.size})
    return result

 
def k12ai_file_md5(file_path):
    m = hashlib.md5()
    with open(file_path,'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            m.update(data)
    return m.hexdigest()
