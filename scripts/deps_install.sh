#!/bin/bash

check=`test -e /etc/exports && echo 1 || echo 0`
if [[ x$check == x0 ]]
then
    sudo apt install -y nfs-common nfs-kernel-server
fi

check=`python3 -c "import ssl" 2>&1`
pyver=`python3 --version | cut -d\  -f2`
if [[ x$check != x ]] || [[ $pyver != "3.6.8" ]]
then
    sudo apt-get install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
    echo "############"
    echo "build the python3.6.8: ./configure --enable-shared --enable-optimizations;  make; sudo make install"
    echo "sudo echo "/usr/local/lib" > python3.6.config; sudo /sbin/ldconfig -v"
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

check=`python3 -c "import flask_cors" 2>&1`
if [[ x$check != x ]]                         
then
    sudo pip3 install flask_cors
else
    echo "flask_cors ok!"
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
    sudo apt install -y libffi-dev
    sudo pip3 install zerorpc
else
    echo "zerorpc ok!"
fi

check=`python3 -c "import docker" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install docker
    sudo apt install -f nvidia-docker2
    sudo gpasswd -a $USER docker; newgrp docker
    sudo systemctl restart docker.service
    echo "##############"
    echo "append in /etc/docker/daemon.json"
    echo "{"
    echo "    \"runtimes\": {"
    echo "       \"nvidia\": {"
    echo "            \"path\": \"nvidia-container-runtime\","
    echo "            \"runtimeArgs\": []",
    echo "        }"
    echo "    },"
    echo "    \"default-runtime\": \"nvidia\","
    echo "    \"registry-mirrors\": [\"https://registry.docker-cn.com\"],"
    echo "    \"insecure-registries\":[\"10.255.0.58:9500\"]"
    echo "}"
    echo ""
    echo "sudo systemctl restart docker.service"
    echo "##############"
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

check=`python3 -c "import _jsonnet" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install jsonnet
else
    echo "jsonnet ok"
fi

check=`python3 -c "import redis" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install redis
else
    echo "redis ok"
fi

check=`which xvfb-run`
if [[ x$check == x ]]
then
    sudo apt install -y xvfb
else
    echo "xvfb ok"
fi

check=`python3 -c "import minio" 2>&1`
if [[ x$check != x ]]
then
    sudo pip3 install minio
else
    echo "redis ok"
fi
