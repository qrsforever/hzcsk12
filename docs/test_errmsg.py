#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback

# sys.path.append("..")

try:
    from k12ai_errmsg import hzcsk12_error_message
except:
    try:
        projdir = os.path.abspath(
                os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "..")
        sys.path.append(projdir)
        from k12ai_errmsg import hzcsk12_error_message
    except:
        def hzcsk12_error_message(code, message=None, detail=None, exc=None, exc_info=None):
            return {'code': 999999}

def f1():
    return hzcsk12_error_message(200, "Test", detail=True)

def f2():
    return f1()

def normal_main():
    print(f2())


def f3():
    raise RuntimeError('Test')

def f4():
    return f3()

def except_main():
    try:
        f4()
    except Exception as err:
        print(hzcsk12_error_message(600, "Test2", exc=True))

if __name__ == "__main__":
    normal_main()
    except_main()
