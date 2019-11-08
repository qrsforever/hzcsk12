#!/bin/bash
#=================================================================
# date: 2019-08-30 15:14:44
# title: build_docker
# author: QRS
#=================================================================

export LANG="en_US.utf8"

DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=$(git describe --tags --always)
URL=$(git config --get remote.origin.url)
COMMIT=$(git rev-parse HEAD | cut -c 1-7)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

REPOSITORY="hzcsai_com/waltzcv"
TAG="0.3.$(git rev-list HEAD | wc -l | awk '{print $1}')"

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

docker build --tag $REPOSITORY:$TAG \
             --build-arg REPOSITORY=$REPOSITORY \
             --build-arg TAG=$TAG \
             --build-arg DATE=$DATE \
             --build-arg VERSION=$VERSION \
             --build-arg URL=$URL \
             --build-arg COMMIT=$COMMIT \
             --build-arg BRANCH=$BRANCH \
             .
