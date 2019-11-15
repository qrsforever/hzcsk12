#!/bin/bash
#=================================================================
# date: 2019-08-31 18:11:04
# title: start_docker
# author: QRS
#=================================================================

CURDIR=`pwd`

DEVPORT=8338
VENDOR=hzcsai_com
PROJECT=k12cv
REPOSITORY="$VENDOR/$PROJECT"

WORKDIR=/hzcsk12/

DATASETSDIR=/data/datasets
PRETRAINDIR=/data/pretrained

### Jupyter
if [[ x$1 == xdev ]]
then
    ROOTDIR=`cd $CURDIR/../..; pwd` 
    if [[ ! -d $ROOTDIR/hzcsnote ]]
    then
        cd $ROOTDIR
        git clone https://gitee.com/hzcsai_com/hzcsnote.git
        cd - > /dev/null
        if [[ ! -d $ROOTDIR/hzcsnote ]]
        then
            echo "Cann't download hzcsnote"
            exit 0
        fi
    fi
    JNAME=${PROJECT}-dev
    check_exist=`docker container ls --filter name=$JNAME --filter status=running -q`
    if [[ x$check_exist == x ]]
    then
        docker run -dit --name $JNAME --restart unless-stopped \
            --runtime nvidia --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
            --volume $DATASETSDIR:$DATASETSDIR \
            --volume $PRETRAINDIR:/root/.cache/torch/checkpoints \
            --volume $CURDIR/cauchy:$WORKDIR/cauchy \
            --volume $CURDIR/app:$WORKDIR/app \
            --volume $ROOTDIR/hzcsnote:$WORKDIR/app/notebook \
            --network host --hostname $PROJECT ${REPOSITORY}-dev \
            /bin/bash -c "umask 0000; jupyter notebook --no-browser --notebook-dir=$WORKDIR/app --allow-root --ip=0.0.0.0 --port=$DEVPORT"
    else
        echo "$JNAME: already run!!!"
    fi
    exit 0
fi

items=($(docker images --filter "label=org.label-schema.name=$REPOSITORY" --format "{{.Repository}}:{{.Tag}}"))
count=${#items[@]}

if (( $count == 0 ))
then
    echo "not found correct image!"
    exit 0
fi

if (( $count > 1 ))
then
    i=0
    while (( i < $count ))
    do
        echo "$i. ${items[$i]}"
        (( i = i + 1 ))
    done
    echo -n "Select: "
    read select
else
    select=0
fi

echo "use image: ${items[$select]}"

container=`docker container  ls --filter ancestor=${items[$select]} --filter status=running -q`

if [[ x$container != x ]]
then
    echo "${items[$select]}: $container already start, you can stop/remove it before start"
    exit 0
fi

containers=(`docker container ls --filter name=$PROJECT --filter status=running -q`)

for c in $containers
do
    echo "stop $c"
    docker container stop $c
done

docker inspect ${items[$select]} --format  '{{json .ContainerConfig.Labels}}' | python -m json.tool

cmd=$(docker inspect ${items[$select]} --format '{{index .ContainerConfig.Labels "org.label-schema.docker.cmd"}}')

if [[ x$cmd != x ]]
then
    $cmd
else
    echo "not found command in org.label-schema.docker.cmd"
fi
