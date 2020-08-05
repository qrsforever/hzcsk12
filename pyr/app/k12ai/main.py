#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess

from k12ai.common.rpc_message import k12ai_send_message

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--pyfile',
            default=None,
            type=str,
            dest='pyfile',
            help="python file")
    args = parser.parse_args()

    if not os.path.exists(args.pyfile):
        print(f'not found pyfile: {args.pyfile}')

    try:
        k12ai_send_message('console', {'status': 'running'})
        runner = subprocess.Popen(
            args=['python', '-u', args.pyfile],
            encoding='utf8',
            bufsize=1,
            stdin=sys.stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        while True:
            output = runner.stdout.readlines()
            if len(output) == 0 and runner.poll() is not None:
                errs = runner.stderr.readlines()
                if len(errs) > 0:
                    k12ai_send_message('console', {'log': ''.join(errs)})
                break
            k12ai_send_message('console', {'log': ''.join(output)})
        k12ai_send_message('console', {'status': 'finish'}, end=True)
    except Exception as err:
        k12ai_send_message('console', {'status': 'error', 'log': f'{err}'}, end=True)
