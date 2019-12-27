#!/bin/bash

cur_fil=${BASH_SOURCE[0]}
cur_dir=`dirname $cur_fil`

jsonnet --ext-str task='sentiment_analysis' $cur_dir/k12nlp.jsonnet
