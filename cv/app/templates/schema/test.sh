#!/bin/bash

cur_fil=${BASH_SOURCE[0]}
cur_dir=`dirname $cur_fil`

task='cls'
network='base_model'
dataset_name='mnist'

case $1 in
    'det'):
        task='det'
        network='vgg16_ssd300'
        dataset_name='voc'
        ;;
    *):
        ;;
esac

jsonnet \
    --ext-str task=$task \
    --ext-str network=$network \
    --ext-str dataset_name=$dataset_name \
    $cur_dir/k12ai_cv.jsonnet
