#!/bin/bash

check=`python3 -c "import ssl" 2>&1`
if [[ x$check != x ]]
then
    echo "############"
    echo "1. sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev"
    echo "2. build the python3.6.8: ./configure --enable-shared --enable-optimizations;  make; sudo make install"
    echo "more see: https://tecadmin.net/install-python-3-6-ubuntu-linuxmint/"
    echo "############"
    exit -1
else
    echo "install python3 with ssl ok!"
fi

check=`python3 -c "import flask" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install flask
else
    echo "flask ok!"
fi

check=`python3 -c "import consul" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install python-consul
else
    echo "consul ok!"
fi

check=`python3 -c "import zerorpc" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install zerorpc
else
    echo "zerorpc ok!"
fi

check=`python3 -c "import docker" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install docker
else
    echo "py docker ok!"
fi

check=`python3 -c "import GPUtil" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install GPUtil
else
    echo "GPUtil ok"
fi

check=`python3 -c "import psutil" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install psutil
else
    echo "psutil ok"
fi

check=`python3 -c "import dictlib" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install dictlib
else
    echo "dictlib ok"
fi

check=`python3 -c "import pyhocon" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install pyhocon
else
    echo "pyhocon ok"
fi

check=`python3 -c "import redis" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install redis
else
    echo "redis ok"
fi
