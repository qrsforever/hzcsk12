#!/bin/bash
#=================================================================
# date: 2019-12-02 14:59:06
# title: k12ai
# author: QRS
# descriptor: called by cron hourly
#=================================================================

cur_fil=${BASH_SOURCE[0]}
top_dir=`cd $(dirname $cur_fil)/..; pwd`

log_fil=/tmp/k12ai_cron.txt

__script_logout()
{
    dt=`date +"%Y-%m-%d %H:%M:%S"`
    echo $dt: $* | tee -a $log_fil
}

__main()
{
    if [[ -f ${log_fil} ]]
    then
        mv ${log_fil} ${log_fil}_bak
    fi
    __script_logout
    $top_dir/scripts/start_services.sh dev nocheck
}

__main
