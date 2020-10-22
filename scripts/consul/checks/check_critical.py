#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import json
import hashlib
import requests
from urllib.parse import quote_plus, unquote_plus

def main():
    consul_stdin = json.load(sys.stdin)
    print(json.dumps(consul_stdin))

if __name__ == "__main__":
    main()

