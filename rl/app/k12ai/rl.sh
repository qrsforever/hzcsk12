#!/bin/bash
#=================================================================
# date: 2020-04-07 14:17:40
# title: rl
# author: QRS
#=================================================================

cur_fil=${BASH_SOURCE[0]}
cur_dir=$(dirname $cur_fil)

xvfb-run -a -s "-screen 0 300x300x24" python ${cur_dir}/main.py $*
