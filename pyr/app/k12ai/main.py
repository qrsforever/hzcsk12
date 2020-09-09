#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import select
import subprocess

from k12ai.common.rpc_message import k12ai_send_message

LINE_WIDTH = 80

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

    # os.environ['PYTHONUNBUFFERED'] = "1"
    try:
        k12ai_send_message('runlog', {
            'status': 'running',
            'log': '#' * LINE_WIDTH + '\n' + '{:^80s}'.format('RUNNING') + '\n' + '#' * LINE_WIDTH + '\n'
        })
        runner = subprocess.Popen(
            args=['python', args.pyfile],
            encoding='utf8',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)

        poll = select.poll()
        poll.register(runner.stdout, select.POLLIN)

        cache = []
        stime = time.time()
        while True:
            if poll.poll(1):
                output = runner.stdout.readline()
                if output == '' and runner.poll() is not None:
                    errs = []
                    for err in runner.stderr.readlines():
                        if all([not err.startswith(x) for x in ('GPU av', 'TPU av', 'CUDA_VI')]):
                            errs.append(err)
                    if len(errs) > 0:
                        k12ai_send_message('runlog', {'log': ''.join(errs)})
                    break
                cache.append(output)
                if time.time() > stime + 0.5:
                    k12ai_send_message('runlog', {'log': ''.join(cache)})
                    cache.clear()
                    stime = time.time()
            else:
                time.sleep(0.5)
        if len(cache) > 0:
            k12ai_send_message('runlog', {'log': ''.join(cache)})
        k12ai_send_message('runlog', {
            'status': 'finished',
            'log': '#' * LINE_WIDTH + '\n' + '{:^80s}'.format('FINISHED') + '\n' + '#' * LINE_WIDTH + '\n'
        }, end=True)
    except Exception as err:
        k12ai_send_message('runlog', {
            'status': 'finished',
            'log': f'{err}.\n'
        }, end=True)
