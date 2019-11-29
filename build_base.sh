#!/bin/bash
#=================================================================
# date: 2019-11-15 09:55:57
# title: build_docker
# author: QRS
#=================================================================

export LANG="en_US.utf8"

DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=$(git describe --tags --always)
URL=$(git config --get remote.origin.url)
COMMIT=$(git rev-parse HEAD | cut -c 1-7)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

VENDOR=hzcsai_com
PROJECT=hzcsk12
REPOSITORY="$VENDOR/$PROJECT"
TAG="1.0.$(git rev-list HEAD | wc -l | awk '{print $1}')"

echo "DATE: $DATE"
echo "VERSION: $VERSION"
echo "URL: $URL"
echo "COMMIT: $COMMIT"
echo "BRANCH: $BRANCH"
echo "IMAGE: $REPOSITORY:$TAG"

echo "build $base_image"
docker build --tag ${REPOSITORY}-base:latest \
    --build-arg DATE=$DATE \
    --build-arg VENDOR=$VENDOR \
    --build-arg PROJECT=$PROJECT \
    --build-arg REPOSITORY=$REPOSITORY \
    --build-arg TAG=$TAG \
    --build-arg VERSION=$VERSION \
    --build-arg URL=$URL \
    --build-arg COMMIT=$COMMIT \
    --build-arg BRANCH=$BRANCH \
    --file Dockerfile.base .
