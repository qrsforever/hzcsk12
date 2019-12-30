#!/bin/bash

cur_fil=${BASH_SOURCE[0]}
cur_dir=`dirname $cur_fil`

jsonnet --ext-str task='sentiment_analysis' --ext-str dataset_path='/data/datasets' --ext-str dataset_name='sst' $cur_dir/helper.libsonnet
