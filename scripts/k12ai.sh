#!/bin/bash
#=================================================================
# date: 2019-12-02 14:59:06
# title: k12ai
# author: QRS
# descriptor: called by cron hourly
#=================================================================

cur_fil=${BASH_SOURCE[0]}
top_dir=`cd $(dirname $cur_fil)/..; pwd`

__main()
{
    result=$(mountpoint /data 2>&1 | grep "is a mountpoint")
    if [[ x$result == x ]]
    then
        if [[ `id -u` == 0 ]]
        then
            mount -t nfs dataserver:/data /data
        else
            sudo mount -t nfs dataserver:/data /data
        fi
    fi
    $top_dir/scripts/start_services.sh dev nocheck
}

__main
