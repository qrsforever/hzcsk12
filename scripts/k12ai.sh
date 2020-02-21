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
    result=$(mountpoint /data2 2>&1 | grep "is a mountpoint")
    if [[ x$result == x ]]
    then
        if [[ `id -u` == 0 ]]
        then
            mount -t nfs dataserver:/data /data2
        else
            sudo mount -t nfs dataserver:/data /data2
        fi
    fi
    $top_dir/scripts/start_services.sh dev nocheck
}

if [[ x$1 == x ]]
then
    __main
else
    if [[ $# > 0 ]] && [[ $1 != 'help' ]]
    then
        if [[ x$1 == xai ]] || [[ x$1 == xml ]] || [[ x$1 == xcv ]] || [[ x$1 == xnlp ]] || [[ x$1 == xrl ]]
        then
            act=$2
            if [[ x$2 == x ]]
            then
                act="restart"
            fi
            if [[ $act == start ]] || [[ $act == stop ]] || [[ $act == restart ]]
            then
                $top_dir/scripts/start_services.sh single $1 $act
            fi
        fi
    else
        echo "k12ai.sh [ai|cv|nlp|rl] [start|stop|restart]"
    fi
fi
