#!/bin/bash
#=================================================================
# date: 2019-12-02 11:36:17
# title: start_docker
# author: QRS
#=================================================================

export LANG="en_US.utf8"

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`
DST_DIR='/hzcsk12'

VENDOR=hzcsai_com

K12AI_PROJECT=k12ai
K12AI_PORT=8118

K12CV_PROJECT=k12cv
K12CV_PORT=8138

K12NLP_PROJECT=k12nlp
K12NLP_PORT=8148

K12RL_PROJECT=k12rl
K12RL_PORT=8158

__start_notebook()
{
    PROJECT=$1
    PORT=$2

    REPOSITORY=$VENDOR/$PROJECT
    JNAME=${PROJECT}-dev

    cd $TOP_DIR
    if [[ ! -d ../hzcsnote ]]
    then
        cd ..
        git clone https://gitee.com/hzcsai_com/hzcsnote.git
        if [[ ! -d hzcsnote ]]
        then
            echo "Cann't download hzcsnote"
            exit 0
        fi
        cd - > /dev/null
    fi
    cd - > /dev/null

    check_exist=`docker container ls --filter name=${JNAME} --filter status=running -q`
    if [[ x$check_exist == x ]]
    then
        subdir=`echo $PROJECT | cut -c4-`
        if [[ ! -d $TOP_DIR/../hzcsnote/$subdir ]]
        then
            subdir=
        fi
        SOURCE_NOTE_DIR=`cd $TOP_DIR/../hzcsnote/$subdir; pwd`
        TARGET_NOTE_DIR=$DST_DIR/hzcsnote
        docker run -dit --name ${JNAME} --restart unless-stopped \
            --volume $SOURCE_NOTE_DIR:$TARGET_NOTE_DIR \
            ${@:3:$#} --network host --hostname ${JNAME} ${REPOSITORY} \
            /bin/bash -c "umask 0000; jupyter notebook --no-browser --notebook-dir=$TARGET_NOTE_DIR --allow-root --ip=0.0.0.0 --port=$PORT"

    else
        echo "$JNAME: already run!!!"
    fi
}

__main()
{
    if [[ x$1 == x ]] || [[ x$1 == xall ]] || [[ x$1 == xai ]]
    then
        __start_notebook $K12AI_PROJECT $K12AI_PORT \
            --env PYTHONPATH=$DST_DIR/hzcsnote \
            --volume $TOP_DIR/cv/app:$DST_DIR/hzcsnote/cv/app \
            --volume $TOP_DIR/nlp/app:$DST_DIR/hzcsnote/nlp/app \
            --volume $TOP_DIR/rl/app:$DST_DIR/hzcsnote/rl/app \
            --volume /data:/data
    fi
    
    if [[ x$1 == xcv ]]
    then
        __start_notebook $K12CV_PROJECT $K12CV_PORT \
            --runtime nvidia --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 \
            --volume $TOP_DIR/cv/app:$DST_DIR/cv/app \
            --volume $TOP_DIR/cv/torchcv:$DST_DIR/cv/torchcv \
            --volume $DST_DIR/cv/torchcv/lib/exts \
            --volume /data:/data
    fi

    if [[ x$1 == xnlp ]]
    then
        __start_notebook $K12NLP_PROJECT $K12NLP_PORT \
            --runtime nvidia --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 \
            --volume $TOP_DIR/nlp/app:$DST_DIR/nlp/app \
            --volume $TOP_DIR/nlp/allennlp/allennlp:$DST_DIR/nlp/allennlp \
            --volume $TOP_DIR/nlp/allennlp-reading-comprehension/allennlp_rc:$DST_DIR/nlp/allennlp_rc \
            --volume /data/nltk_data:/root/nltk_data \
            --volume /data:/data
    fi

    if [[ x$1 == xrl ]]
    then
        __start_notebook $K12RL_PROJECT $K12RL_PORT \
            --runtime nvidia --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 \
            --volume $TOP_DIR/rl/app:$DST_DIR/rl/app \
            --volume $TOP_DIR/rl/rlpyt:$DST_DIR/rl/rlpyt\
            --volume /data:/data
    fi
}

__main $*
