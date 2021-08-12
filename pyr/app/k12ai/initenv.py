#!/usr/bin/python3
# -*- coding: utf-8 -*-

import io

from PIL import Image
from k12ai.common.rpc_message import k12ai_send_message as send_msg
from k12ai.common.util_misc import img2b64


def pyr_status(status):
    send_msg('error', {
        'status': status,
    })
    send_msg('runlog', {
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
