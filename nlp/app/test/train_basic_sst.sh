#!/bin/bash

script_dir=`dirname ${BASH_SOURCE[0]}`
script_dir=`cd $script_dir; pwd`
app_dir=`dirname $script_dir`
tmp_dir=/data/tmp

__run()
{
    echo $@
    $@
}

cd $app_dir


if [[ -d $tmp_dir/basic_sst ]]
then
    flags='--recover'
fi
__run allennlp train config/basic_sst.jsonnet --serialization-dir $tmp_dir/basic_sst --include-package hzcsnlp $flags


cd - >/dev/null
