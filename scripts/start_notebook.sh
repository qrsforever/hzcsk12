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
K12NB_WWW_PORT=9091

K12PYR_PROJECT=k12pyr
K12PYR_PORT=8178

__start_notebook()
{
    PROJECT=$1
    PORT=$2
    REPOSITORY=$VENDOR/$PROJECT

    if [[ $PROJECT == $K12NB_PROJECT ]]
    then
        JNAME=k12nb
    else
        JNAME=$PROJECT-nb
    fi

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

    echo "###${JNAME}"
    check_exist=`docker container ls --filter name=${JNAME} --filter status=running -q`
    echo "docker container ls --filter name=${JNAME} --filter status=running -q"
    if [[ x$check_exist == x ]]
    then
        subdir=`echo $PROJECT | cut -c4-`
        if [[ ! -d $TOP_DIR/../hzcsnote/$subdir ]]
        then
            subdir=
        fi
        SOURCE_NOTE_DIR=`cd $TOP_DIR/../hzcsnote/$subdir; pwd`
        TARGET_NOTE_DIR=$DST_DIR/hzcsnote
        if [[ $PROJECT == $K12NB_PROJECT ]]
        then
            echo "start http.sever"
            cd $SOURCE_NOTE_DIR/k12libs/www
            ./run.sh
            cd - >/dev/null
            xvfb_args="xvfb-run -a -s \"-screen 0 1400x900x24\""
        else
            xvfb_args=""
        fi
        docker run -dit --name ${JNAME} --restart unless-stopped --shm-size 10g \
            --volume $SOURCE_NOTE_DIR:$TARGET_NOTE_DIR \
            --entrypoint /bin/bash ${@:3:$#} --network host --hostname ${JNAME} ${REPOSITORY} \
            -c "umask 0000; $xvfb_args jupyter notebook --no-browser --notebook-dir=$TARGET_NOTE_DIR --allow-root --ip=0.0.0.0 --port=$PORT"

    else
        echo "$JNAME: already run!!!"
    fi
}

__main()
{
    if [[ x$1 == xpyr ]]
    then
        __start_notebook $K12PYR_PROJECT $K12PYR_PORT \
            --env PYTHONPATH=$DST_DIR:$DST_DIR/pyr/app:$DST_DIR/pyr/pytorch-lightning \
            --volume $TOP_DIR/pyr/app:$DST_DIR/pyr/app \
            --volume $TOP_DIR/pyr/pytorch-lightning/pytorch_lightning:$DST_DIR/pyr/pytorch-lightning/pytorch_lightning \
            --volume /data:/data \
            --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints
    else
        __start_notebook $K12NB_PROJECT $K12NB_PORT \
            --env PYTHONPATH=$DST_DIR/hzcsnote \
            --volume $TOP_DIR/ml/app:$DST_DIR/hzcsnote/ml/app \
            --volume $TOP_DIR/cv/app:$DST_DIR/hzcsnote/cv/app \
            --volume $TOP_DIR/nlp/app:$DST_DIR/hzcsnote/nlp/app \
            --volume $TOP_DIR/rl/app:$DST_DIR/hzcsnote/rl/app \
            --volume $TOP_DIR/3d/app:$DST_DIR/hzcsnote/3d/app \
            --volume $TOP_DIR/pyr/app:$DST_DIR/hzcsnote/pyr/app \
            --volume $TOP_DIR/services/k12ai:$DST_DIR/hzcsnote/k12libs/k12ai \
            --volume /data:/data \
            --volume /raceai:/raceai \
            --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints
    fi
}

__main $*
