#!/bin/bash

cur_fil=${BASH_SOURCE[0]}
cur_dir=`dirname $cur_fil`

if [[ x$1 == x ]]
then
    jsonnet --ext-str task='cls' --ext-str dataset_path='/data/datasets/cv' --ext-str dataset_name='mnist' $cur_dir/k12ai_cv.jsonnet
else
    jsonnet --ext-str task='cls' --ext-str dataset_path='/data/datasets/cv' --ext-str dataset_name='mnist' $1
fi
