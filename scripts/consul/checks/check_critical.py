#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import json
import hashlib
from urllib.parse import quote_plus, unquote_plus
from urllib.request import urlopen

def main():
    consul_stdin = json.load(sys.stdin)
    data = json.dumps(consul_stdin)
    print(data)
    with open('/tmp/k12check.txt', 'w+') as fw:
        fw.write(data)

if __name__ == "__main__":
    main()

