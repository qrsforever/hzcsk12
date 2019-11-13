#!/bin/bash

script_dir=`dirname ${BASH_SOURCE[0]}`
script_dir=`cd $script_dir; pwd`
app_dir=`dirname $script_dir`

__run()
{
    echo $@
    $@
}

cd $app_dir


if [[ -d tmp/callback_basic_sst ]]
then
    flags='--recover'
fi
__run allennlp train config/callback_basic_sst.jsonnet --serialization-dir tmp/callback_basic_sst --include-package hzcsnlp $flags


cd - >/dev/null
