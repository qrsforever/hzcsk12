#!/bin/bash

cur_fil=${BASH_SOURCE[0]}
cur_dir=`dirname $cur_fil`

if [[ x$1 == x ]]
then
    jsonnet --ext-str task='sentiment_analysis' --ext-str dataset_path='/data/datasets/nlp' --ext-str dataset_name='sst' $cur_dir/k12ai_nlp.jsonnet
else
    jsonnet --ext-str task='sentiment_analysis' --ext-str dataset_path='/data/datasets/nlp' --ext-str dataset_name='sst' $1
fi
