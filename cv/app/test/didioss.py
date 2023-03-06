#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from minio import Minio


mc = Minio(
    endpoint='s3.didiyunapi.com',
    access_key='AKDD002E38WR1J7RMPTGRIGNVCVINY',
    secret_key='ASDDXYWs45ov7MNJbj5Wc2PM9gC0FSqCIkiyQkVC',
    secure=False)

def _object_list(client, prefix, recurive=False, bucket_name='scene-dataset'):
    if bucket_name is None:
        bucket_name = 'scene-dataset'
    return list(client.list_objects(bucket_name, prefix=prefix, recursive=recurive))


def _object_get(client, remote_path, bucket_name=None, prefix_map=None):
    if bucket_name is None:
        bucket_name = 'scene-dataset'
    remote_path = remote_path.lstrip(os.path.sep)

    if prefix_map:
        if prefix_map[0][0] == '/':
            prefix_map[0] = prefix_map[0][1:]
        if prefix_map[1][0] == '/':
            prefix_map[1] = prefix_map[1][1:]

    result = []
    for obj in _object_list(client, remote_path, True, bucket_name):
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

print(_object_get(mc, 'cv/', bucket_name='scene-dataset'))
