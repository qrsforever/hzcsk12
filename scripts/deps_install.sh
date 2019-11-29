#!/bin/bash

check=`python3 -c "import consul"`
if [[ x$check != x ]]
then
    pip3 install consul --user
else
    echo "consul ok!"
fi

check=`python3 -c "import zerorpc"`
if [[ x$check != x ]]
then
    pip3 install zerorpc --user
else
    echo "zerorpc ok!"
fi

check=`python3 -c "import docker"`
if [[ x$check != x ]]
then
    pip3 install docker-py --user
else
    echo "docker-py ok!"
fi

check=`python3 -c "import GPUtil"`
if [[ x$check != x ]]
then
    pip3 install GPUtil --user
else
    echo "GPUtil ok"
fi

check=`python3 -c "import psutil"`
if [[ x$check != x ]]
then
    pip3 install psutil --user
else
    echo "psutil ok"
fi
