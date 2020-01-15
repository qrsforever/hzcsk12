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


if [[ -d $tmp_dir/callback_bidaf_squad ]]
then
    flags='--recover'
fi
__run allennlp train /hzcsk12/nlp/app/test/callback_bidaf_squad.jsonnet --serialization-dir $tmp_dir/callback_bidaf_squad --include-package allennlp_rc $flags


cd - >/dev/null
