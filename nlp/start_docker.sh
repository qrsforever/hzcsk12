#!/bin/bash
#=================================================================
# date: 2019-11-05 12:00:26
# title: start_docker
# author: QRS
#=================================================================

export LANG="en_US.utf8"

CURDIR=`pwd`

MAJOR=1
MINOR=0
PORT=8349
DEVPORT=8348

DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=$(git describe --tags --always)
URL=$(git config --get remote.origin.url)
COMMIT=$(git rev-parse HEAD | cut -c 1-7)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

VENDOR=hzcsai_com
PROJECT=k12nlp
REPOSITORY="$VENDOR/$PROJECT"
TAG="$MAJOR.$MINOR.$(git rev-list HEAD | wc -l | awk '{print $1}')"

WORKDIR=/hzcsk12
DATADIR=/data

### Jupyter
if [[ x$1 == xdev ]]
then
    check_exist=`docker images $REPOSITORY-dev -q`
    if [[ x$check_exist == x ]]
    then
        if [[ ! -d .jupyter_config ]]
        then
            git clone https://gitee.com/lidongai/jupyter_config.git .jupyter_config
        fi
        sed "s/{{REPLACEME}}/${VENDOR}\/$PROJECT:$TAG/g" Dockerfile.dev > .Dockerfile.dev
        docker build --tag ${REPOSITORY}-dev --file .Dockerfile.dev .
    fi
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
            --volume $DATADIR:$DATADIR \
            --volume $CURDIR/allennlp:$WORKDIR/allennlp \
            --volume $CURDIR/app:$WORKDIR/app \
            --volume $ROOTDIR/hzcsnote:$WORKDIR/app/notebook \
            --network host ${REPOSITORY}-dev \
            /bin/bash -c "umask 0000; jupyter notebook --no-browser --notebook-dir=$WORKDIR/app --allow-root --ip=0.0.0.0 --port=$DEVPORT"
    else
        echo "$JNAME: already run!!!"
    fi
    exit 0
fi

__build_image()
{
    echo "build $REPOSITORY:$TAG"
    docker build --tag $REPOSITORY:$TAG \
        --build-arg VENDOR=$VENDOR \
        --build-arg PROJECT=$PROJECT \
        --build-arg REPOSITORY=$REPOSITORY \
        --build-arg TAG=$TAG \
        --build-arg DATE=$DATE \
        --build-arg VERSION=$VERSION \
        --build-arg URL=$URL \
        --build-arg COMMIT=$COMMIT \
        --build-arg BRANCH=$BRANCH \
        --build-arg PORT=$PORT \
        --file Dockerfile.nlp .
}

items=($(docker images --filter "label=org.label-schema.name=$REPOSITORY" --format "{{.Tag}}"))
count=${#items[@]}

if (( $count == 0 ))
then
    __build_image
    imageName=`docker images $REPOSITORY:$TAG --format "{{.Repository}}:{{.Tag}}"`
else
    lastest=0
    i=0
    while (( i < $count ))
    do
        echo "$i. $REPOSITORY:${items[$i]}"
        if [[ $lastest != 1 ]] && [[ $(echo ${items[$i]} | cut -d \. -f1-2) == $MAJOR.$MINOR ]]
        then
            lastest=1
        fi
        (( i = i + 1 ))
    done
    if (( $lastest == 0 ))
    then
        echo "$i. $REPOSITORY:$TAG (need build)"
        echo -n "Select: "
        read select
        if [[ x$i == x$select ]]
        then
            __build_image
            imageName=`docker images $REPOSITORY:$TAG --format "{{.Repository}}:{{.Tag}}"`
        else
            imageName=$REPOSITORY:${items[$select]}
        fi
    else
        if (( $i > 1 ))
        then
            echo -n "Select: "
            read select
            imageName=$REPOSITORY:${items[$select]}
        else
            imageName=$REPOSITORY:${items[0]}
        fi
    fi
fi

echo "use image: $imageName"

if [[ x$imageName == x ]]
then
    echo "Image is null"
    exit 0
fi

container=`docker container  ls --filter ancestor=$imageName --filter status=running -q`

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

docker inspect ${imageName} --format '{{json .ContainerConfig.Labels}}' | python -m json.tool

cmd=$(docker inspect ${imageName} --format '{{index .ContainerConfig.Labels "org.label-schema.docker.cmd"}}')

if [[ x$cmd != x ]]
then
    echo "run start..."
    $cmd
else
    echo "not found command in org.label-schema.docker.cmd"
fi
