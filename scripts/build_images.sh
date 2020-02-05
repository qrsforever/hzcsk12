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
MINOR_K12AI=1

MAJOR_K12CV=1
MINOR_K12CV=1

MAJOR_K12NLP=1
MINOR_K12NLP=1

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

    DESTDIR=$(dirname $4)
    DOCKERFILE=$(basename $4)

    TAG=$MAJOR.$MINOR.$NUMBER
    REPOSITORY=$VENDOR/$PROJECT

    build_flag=0

    force=$5
    if [[ x$force == x ]]
    then
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
    else
        build_flag=1
    fi
    if (( build_flag == 1 ))
    then
        echo "build image: $REPOSITORY:$TAG"

        cd $TOP_DIR/$DESTDIR

        if [[ $PROJECT == "k12ai" ]] && [[ ! -d .jupyter_config ]]
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

        if [[ $? != 0 ]]
        then
            echo "docker build $REPOSITORY:$TAG fail"
            exit $?
        fi
        docker tag $REPOSITORY:$TAG $REPOSITORY
        cd - >/dev/null
    else
        echo "No need build new image $REPOSITORY!"
    fi
}

__main()
{
    image='all'
    if [[ x$1 == xai ]] || [[ x$1 == xcv ]] || [[ x$1 == xnlp ]] || [[ x$1 == xrl ]]
    then
        image=$1
        shift
    fi
    force=0
    if [[ x$1 == x1 ]]
    then
        force=1
    fi
    if [[ x$image == xall ]]
    then
        __build_image "k12ai" $MAJOR_K12AI $MINOR_K12AI Dockerfile.ai $force
        __build_image "k12cv" $MAJOR_K12CV $MINOR_K12CV cv/Dockerfile.cv $force
        __build_image "k12nlp" $MAJOR_K12NLP $MINOR_K12NLP nlp/Dockerfile.nlp $force
        __build_image "k12rl" $MAJOR_K12NLP $MINOR_K12NLP rl/Dockerfile.rl $force
    else
        __build_image "k12$image" $MAJOR_K12NLP $MINOR_K12NLP $image/Dockerfile.$image $force
    fi
}

__main $@
