#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import re
import os
import sys
import traceback

from initenv import pyr_print as print # noqa
from initenv import pyr_error, pyr_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--pyfile',
            default='/cache/pyrcode.py',
            type=str,
            dest='pyfile',
            help="python file")
    args = parser.parse_args()

    if os.path.exists(args.pyfile):
        with open(args.pyfile) as fr:
            try:
                pyr_status('running')
                exec(compile(fr.read(), 'pyrcode', 'exec'), globals())
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                errinfo = {
                    'err_type': exc_type.__name__,
                    'err_text': str(exc_value),
                }
                errinfo['trackback'] = []
                tbs = traceback.extract_tb(exc_tb)
                res = re.search(r'\(pyrcode, line (?P<line>\d+)\)', errinfo['err_text'])
                if res:
                    errinfo['trackback'].append({
                        'filename': 'pyrcode',
                        'linenum': int(res.groupdict()['line'])
                    })
                else:
                    for tb in tbs:
                        if 'pyrcode' != tb.filename:
                            continue
                        err = {
                            'filename': tb.filename,
                            'linenum': tb.lineno,
                        }
                        errinfo['trackback'].append(err)
                pyr_error(errinfo)
            finally:
                pyr_status('finished')
    else:
        pyr_error({'err_type': 'FileNotFoundError', 'err_text': 'not found %s' % args.pyfile})
