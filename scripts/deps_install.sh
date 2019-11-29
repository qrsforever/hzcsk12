#!/bin/bash

check=`python3 -c "import ssl" 2>&1`
if [[ x$check != x ]]
then
    echo "############"
    echo "1. sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev"
    echo "2. build the source3.6.8: ./configure --prefix=/usr/local/python3.6; make; sudo make install"
    echo "3. set env in bashrc: both PATH and LD_LIBRARY_PATH"
    echo "more see: https://tecadmin.net/install-python-3-6-ubuntu-linuxmint/"
    echo "############"
    exit -1
else
    echo "install python3 with ssl ok!"
fi

check=`python3 -c "import flask" 2>&1`
if [[ x$check != x ]]
then
    pip3 install flask --user
else
    echo "flask ok!"
fi

check=`python3 -c "import consul" 2>&1`
if [[ x$check != x ]]
then
    pip3 install python-consul --user
else
    echo "consul ok!"
fi

check=`python3 -c "import zerorpc" 2>&1`
if [[ x$check != x ]]
then
    pip3 install zerorpc --user
else
    echo "zerorpc ok!"
fi

check=`python3 -c "import docker" 2>&1`
if [[ x$check != x ]]
then
    pip3 install docker --user
else
    echo "py docker ok!"
fi

check=`python3 -c "import GPUtil" 2>&1`
if [[ x$check != x ]]
then
    pip3 install GPUtil --user
else
    echo "GPUtil ok"
fi

check=`python3 -c "import psutil" 2>&1`
if [[ x$check != x ]]
then
    pip3 install psutil --user
else
    echo "psutil ok"
fi
