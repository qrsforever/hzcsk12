#!/usr/bin/python3
# -*- coding: utf-8 -*-

import io
from PIL import Image
from k12ai.common.rpc_message import k12ai_send_message as send_msg
from k12ai.common.util_misc import img2b64


def pyr_imshow(img, *args, **kwargs):
    send_msg('runlog', {
        'status': 'running',
        'imshow': img2b64(img)
    })


def pyr_print(*args, **kwargs):
    with io.StringIO() as sio:
        print(*args, **kwargs, file=sio)
        send_msg('runlog', {
            'status': 'running',
            'log': sio.getvalue()
        })


Image._show = pyr_imshow
