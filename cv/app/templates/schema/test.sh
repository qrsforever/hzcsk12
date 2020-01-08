#!/bin/bash

cur_fil=${BASH_SOURCE[0]}
cur_dir=`dirname $cur_fil`

if [[ x$1 == x ]]
then
    jsonnet \
        --ext-str task='cls' \
        --ext-str dataset_name='mnist' \
        --ext-str dataset_root='/hzcsk12/cv/data/datasets' \
        --ext-str checkpt_root='/hzcsk12/cv/data/cache' \
        --ext-str pretrained_path='/hzcsk12/cv/data/pretrained' \
        $cur_dir/k12ai_cv.jsonnet
else
    jsonnet \
        --ext-str task='cls' \
        --ext-str dataset_name='mnist' \
        --ext-str dataset_root='/hzcsk12/cv/data/datasets' \
        --ext-str checkpt_root='/hzcsk12/cv/data/cache' \
        --ext-str pretrained_path='/hzcsk12/cv/data/pretrained' \
        $1
fi
