#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file md5_.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-02-20 17:31


import sys
import hashlib
 
def get_file_md5(file_name):
    m = hashlib.md5()
    with open(file_name,'rb') as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data)
 
    return m.hexdigest()


if __name__ == "__main__":
    md5 = get_file_md5(sys.argv[1])
    print(type(md5), md5)
