#!/bin/bash

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
