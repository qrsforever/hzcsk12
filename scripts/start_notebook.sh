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

K12NB_PROJECT=k12nb
K12NB_PORT=8118

__start_notebook()
{
    PROJECT=$1
    PORT=$2

    REPOSITORY=$VENDOR/$PROJECT
    JNAME=k12nb

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
        docker run -dit --name ${JNAME} --restart unless-stopped --shm-size 10g \
            --volume $SOURCE_NOTE_DIR:$TARGET_NOTE_DIR \
            --entrypoint /bin/bash ${@:3:$#} --network host --hostname ${JNAME} ${REPOSITORY} \
            -c "umask 0000; xvfb-run -a -s \"-screen 0 1400x900x24\" jupyter notebook --no-browser --notebook-dir=$TARGET_NOTE_DIR --allow-root --ip=0.0.0.0 --port=$PORT"

    else
        echo "$JNAME: already run!!!"
    fi
}

__main()
{
    __start_notebook $K12NB_PROJECT $K12NB_PORT \
        --env PYTHONPATH=$DST_DIR/hzcsnote:$DST_DIR/hzcsnote/k12libs \
        --volume $TOP_DIR/ml/app:$DST_DIR/hzcsnote/ml/app \
        --volume $TOP_DIR/cv/app:$DST_DIR/hzcsnote/cv/app \
        --volume $TOP_DIR/nlp/app:$DST_DIR/hzcsnote/nlp/app \
        --volume $TOP_DIR/rl/app:$DST_DIR/hzcsnote/rl/app \
        --volume $TOP_DIR/3d/app:$DST_DIR/hzcsnote/3d/app \
        --volume $TOP_DIR/services/k12ai:$DST_DIR/hzcsnote/k12libs/k12ai \
        --volume /data:/data \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints
}

__main $*
