#!/usr/bin/python3
# -*- coding: utf-8 -*-

import docker
from threading import Thread
import time

# print("start1")
# 
# client.containers.run('hzcsai_com/k12nlp-dev', command='allennlp configure',
#     auto_remove=True,
#     detach=True,
#     hostname='gamma',
#     name='k12nlp-test',
#     network_mode='host',
#     remove=True,
#     runtime='nvidia',
#     shm_size='2g',
#     volumes={'/data':{'bind':'/data', 'mode':'rw'}})
# 
# print("start2")


def __container_run(img, cmd, name, **kwargs):
    print(img, cmd, name)
    print(kwargs)
    client.containers.run(img, cmd, name=name, **kwargs)
    print("end thread")

def main():
    client = docker.from_env()
    args = {
            'auto_remove':True,
            'detach':True,
            'hostname':'gamma',
            'network_mode':'host',
            'remove':True,
            'runtime':'nvidia',
            'shm_size':'2g',
            'volumes':{
                '/data':{'bind':'/data', 'mode':'rw'}
                }
            }
    img = 'hzcsai_com/k12nlp-dev'
    cmd = 'allennlp configure' 
    name = 'k12nlp-test'
    Thread(target=lambda **kwargs: client.containers.run(img, cmd, name=name, **{
            'auto_remove':True,
            'detach':True,
            'hostname':'gamma',
            'network_mode':'host',
            'remove':True,
            'runtime':'nvidia',
            'shm_size':'2g',
            'volumes':{
                '/data':{'bind':'/data', 'mode':'rw'}
                }
            }),
        daemon=True, kwargs=args).start()

    time.sleep(5)

    client = docker.from_env()
    con = client.containers.get(name)
    try:
        print("stop1")
        con.stop()
        print("stop2")
    except Exception as err:
        print(err)

    print("end!")

if __name__ == "__main__":
    main()
