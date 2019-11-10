#!/bin/bash
#=================================================================
# date: 2019-11-05 12:00:26
# title: start_docker
# author: QRS
#=================================================================

CURDIR=`pwd`

DEVPORT=8348
VENDOR=hzcsai_com
PROJECT=k12nlp
REPOSITORY="$VENDOR/$PROJECT"

WORKDIR=/hzcsk12/nlp

NBDIR=$CURDIR/jupyter/notebook
DBDIR=/data/datasets/nlp

### Jupyter
if [[ x$1 == xdev ]]
then
    JNAME=${PROJECT}-dev
    check_exist=`docker container ls --filter name=$JNAME --filter status=running -q`
    if [[ x$check_exist == x ]]
    then
        if [[ ! -d $NBDIR ]]
        then
            mkdir -p $NBDIR
        fi
        docker run -dit --name $JNAME --restart unless-stopped \
            --runtime nvidia --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
            --env WORKDIR=$WORKDIR --volume $NBDIR:/notebook --volume $DBDIR:$DBDIR \
            --volume ${CURDIR}/app:$WORKDIR/app --volume ${CURDIR}/allennlp:$WORKDIR/allennlp \
            --network host ${REPOSITORY}-dev \
            /bin/bash -c "umask 0000; jupyter notebook --no-browser --notebook-dir=/notebook --allow-root --ip=0.0.0.0 --port=$DEVPORT"
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
    echo "run start..."
    # $cmd
else
    echo "not found command in org.label-schema.docker.cmd"
fi
