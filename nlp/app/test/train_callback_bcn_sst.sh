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


if [[ -d $tmp_dir/callback_bcn_sst ]]
then
    flags='--recover'
fi
__run allennlp train callback_bcn_sst.jsonnet --serialization-dir $tmp_dir/callback_bcn_sst --include-package hzcsnlp $flags


cd - >/dev/null
