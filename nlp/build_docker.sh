#!/bin/bash
#=================================================================
# date: 2019-11-05
# title: build_docker
# author: QRS
#=================================================================

export LANG="en_US.utf8"

PORT=8349
DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=$(git describe --tags --always)
URL=$(git config --get remote.origin.url)
COMMIT=$(git rev-parse HEAD | cut -c 1-7)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

VENDOR=hzcsai_com
PROJECT=k12nlp
REPOSITORY="$VENDOR/$PROJECT"
TAG="0.4.$(git rev-list HEAD | wc -l | awk '{print $1}')"

echo "DATE: $DATE"
echo "VERSION: $VERSION"
echo "URL: $URL"
echo "COMMIT: $COMMIT"
echo "BRANCH: $BRANCH"
echo "IMAGE: $REPOSITORY:$TAG"

base_image=${REPOSITORY}-base
base_tag=`docker images -q $base_image:latest`

if [[ x$base_tag == x ]]
then
     echo "build $base_image"
     docker build --tag $base_image:latest \
                  --build-arg DATE=$DATE \
                  --file Dockerfile.base .
fi

check_exist=`docker images -q $REPOSITORY:$TAG`

if [[ x$check_exist == x ]]
then
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
                 .
fi

check_ok=`docker images -q $REPOSITORY:$TAG`

if [[ x$check_ok != x ]]
then
    echo "Build Success!"
    if [[ ! -d .jupyter_config ]]
    then
        git clone https://gitee.com/lidongai/jupyter_config.git .jupyter_config
    fi
    sed "s/{{REPLACEME}}/${VENDOR}\/$PROJECT:$TAG/g" Dockerfile.dev > .Dockerfile.dev
    docker build --tag ${REPOSITORY}-dev --file .Dockerfile.dev .
else
    echo "Build Failed!"
fi
