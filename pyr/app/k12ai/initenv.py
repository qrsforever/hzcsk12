#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file initenv.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-08-13


import os
import zerorpc
import io

from PIL import Image
import numpy as np

_RPCClient = None
_RPCEnable = -1
K12AI_DEVELOPER, K12AI_APPID, K12AI_TOKEN, K12AI_OP, K12AI_USER, K12AI_UUID = False, None, None, None, None, None


def send_msg(msgtype, message, end=False):
    global _RPCClient, _RPCEnable, K12AI_DEVELOPER, K12AI_APPID, K12AI_TOKEN, K12AI_OP, K12AI_USER, K12AI_UUID

    if _RPCEnable == 0:
        return

    if _RPCEnable == -1:
        host = os.environ.get('K12AI_RPC_HOST', None)
        port = os.environ.get('K12AI_RPC_PORT', None)
        if not host or not port:
            _RPCEnable = 0
            return

        K12AI_DEVELOPER = True if os.environ.get('K12AI_DEVELOPER', None) else False

        K12AI_APPID = os.environ.get('K12AI_APPID', 'Unkown')
        K12AI_TOKEN = os.environ.get('K12AI_TOKEN', 'Unkown')
        K12AI_OP = os.environ.get('K12AI_OP', 'Unkown')
        K12AI_USER = os.environ.get('K12AI_USER', 'Unkown')
        K12AI_UUID = os.environ.get('K12AI_UUID', 'Unkown')
        _RPCClient = zerorpc.Client(
                connect_to='tcp://{}:{}'.format(host, port),  # noqa
                timeout=3,
                passive_heartbeat=True)
        _RPCEnable = 1

    try:
        if K12AI_DEVELOPER:
            print(message)

        if message:
            _RPCClient.send_message(K12AI_APPID, K12AI_TOKEN, K12AI_OP, K12AI_USER, K12AI_UUID, msgtype, message)
        if end:
            _RPCClient.close()

    except Exception as err:
        if K12AI_DEVELOPER:
            print(err)

def image2bytes(image, width=None, height=None):
    if isinstance(image, bytes):
        return image

    if isinstance(image, str):
        if os.path.isfile(image):
            image = Image.open(image).convert("RGB")
        else:
            raise RuntimeError

    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image.astype('uint8')).convert('RGB')

    if isinstance(image, Image.Image):
        if width and height:
            image = image.resize((width, height))
        bio = io.BytesIO()
        image.save(bio, "PNG")
        bio.seek(0)
        return bio.read()

    raise NotImplementedError(type(image))


def img2b64(x):
    return base64.b64encode(image2bytes(x)).decode()


def pyr_status(status):
    send_msg('error', {
        'status': status,
    })


def pyr_print(*args, **kwargs):
    with io.StringIO() as sio:
        print(*args, **kwargs, file=sio)
        send_msg('runlog', {
            'log': sio.getvalue()
        })


def pyr_error(errinfo):
    send_msg('error', {
        'errinfo': errinfo
    })


def pyr_imshow(img, *args, **kwargs):
    send_msg('runlog', {
        'imshow': img2b64(img)
    })


Image._show = pyr_imshow
