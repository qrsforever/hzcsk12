#!/bin/bash

cur_fil=${BASH_SOURCE[0]}
cur_dir=`dirname $cur_fil`

task='sentiment_analysis'
network='basic_classifier'
dataset_name='sst'

case $1 in
    *):
        ;;
esac

jsonnet \
    --ext-str task=$task \
    --ext-str network=$network \
    --ext-str dataset_name=$dataset_name \
    $cur_dir/k12ai_nlp.jsonnet
