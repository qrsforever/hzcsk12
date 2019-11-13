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


if [[ -d tmp/basic_sst ]]
then
    flags='--recover'
fi
__run allennlp train config/basic_sst.jsonnet --serialization-dir tmp/basic_sst --include-package hzcsnlp $flags


cd - >/dev/null
