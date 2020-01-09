#!/bin/bash

cur_fil=${BASH_SOURCE[0]}
cur_dir=`dirname $cur_fil`

if [[ x$1 == x ]]
then
    jsonnet \
        --ext-str task='cls' \
        --ext-str dataset_name='mnist' \
        --ext-str dataset_root='/datasets' \
        --ext-str checkpt_root='/cache' \
        --ext-str pretrained_path='/pretrained' \
        $cur_dir/k12ai_cv.jsonnet
else
    jsonnet \
        --ext-str task='cls' \
        --ext-str dataset_name='mnist' \
        --ext-str dataset_root='/datasets' \
        --ext-str checkpt_root='/cache' \
        --ext-str pretrained_path='/pretrained' \
        $1
fi
