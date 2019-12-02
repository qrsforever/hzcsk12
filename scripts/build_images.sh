#!/bin/bash
#=================================================================
# date: 2019-12-02 12:43:39
# title: build_images
# author: QRS
#=================================================================

export LANG="en_US.utf8"

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com

MAJOR_K12AI=1
MINOR_K12AI=0

MAJOR_K12CV=1
MINOR_K12CV=0

MAJOR_K12NLP=1
MINOR_K12NLP=0

DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=$(git describe --tags --always)
URL=$(git config --get remote.origin.url)
COMMIT=$(git rev-parse HEAD | cut -c 1-7)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
NUMBER=$(git rev-list HEAD | wc -l | awk '{print $1}')

echo "DATE: $DATE"
echo "VERSION: $VERSION"
echo "URL: $URL"
echo "COMMIT: $COMMIT"
echo "BRANCH: $BRANCH"
echo "NUMBER: $NUMBER"

__build_image()
{
    PROJECT=$1
    MAJOR=$2
    MINOR=$3
    DOCKERFILE=$4

    TAG=$MAJOR.$MINOR.$NUMBER
    REPOSITORY=$VENDOR/$PROJECT

    build_flag=0
    items=($(docker images --filter "label=org.label-schema.name=$REPOSITORY" --format "{{.Tag}}"))
    count=${#items[@]}
    if (( $count == 0 ))
    then
        build_flag=1
    else
        lastest=0
        i=0
        echo "Already exist images:"
        while (( i < $count ))
        do
            echo -e "\t$(expr 1 + $i). $REPOSITORY:${items[$i]}"
            if [[ $lastest != 1 ]] && [[ $(echo ${items[$i]} | cut -d \. -f1-2) == $MAJOR.$MINOR ]]
            then
                lastest=1
            fi
            (( i = i + 1 ))
        done
        if (( $lastest == 0 ))
        then
            echo -ne "\nBuild new image: $REPOSITORY:$TAG (y/N): "
            read result
            if [[ x$result == xy ]] || [[ x$result == xY ]]
            then
                build_flag=1
            fi
        fi
    fi
    if (( build_flag == 1 ))
    then
        echo "build image: $REPOSITORY:$TAG"
        cd $TOP_DIR
        if [[ ! -d .jupyter_config ]]
        then
            git clone https://gitee.com/lidongai/jupyter_config.git .jupyter_config
        fi
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
            --file $DOCKERFILE .
        docker tag $REPOSITORY:$TAG $REPOSITORY
        cd - >/dev/null
    else
        echo "No need build new image $REPOSITORY!"
    fi
}

__main()
{
    __build_image "k12ai" $MAJOR_K12AI $MINOR_K12AI Dockerfile.ai
    __build_image "k12cv" $MAJOR_K12CV $MINOR_K12CV Dockerfile.cv
    __build_image "k12nlp" $MAJOR_K12NLP $MINOR_K12NLP Dockerfile.nlp
}

__main
