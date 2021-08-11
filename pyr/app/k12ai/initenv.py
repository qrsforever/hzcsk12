#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PIL import Image
import time

def pry_imshow(img, *args, **kwargs):
    for _ in range(20):
        print('hello world', flush=True)
        time.sleep(1)

Image._show = pry_imshow
