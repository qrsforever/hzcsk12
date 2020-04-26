#!/bin/bash
#=================================================================
# date: 2019-12-02 14:59:06
# title: k12ai
# author: QRS
# descriptor: called by cron hourly
#=================================================================

cur_fil=${BASH_SOURCE[0]}
top_dir=`cd $(dirname $cur_fil)/..; pwd`

export PYTHONPATH=/home/lidong/.local/lib/python3.6/site-packages:$PYTHONPATH

# /etc/crontab
# 01 *  * * *   root  cd / && /home/lidong/workspace/codes/hzcsai_com/hzcsk12/scripts/k12ai.sh
# @reboot  root  cd / && /home/lidong/workspace/codes/hzcsai_com/hzcsk12/scripts/k12ai.sh


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
    $top_dir/scripts/start_services.sh dev all start "bg"
}

if [[ x$1 == x ]]
then
    __main
else
    if [[ $# > 0 ]] && [[ $1 != 'help' ]]
    then
        act=$2
        if [[ x$2 == x ]]
        then
            act="restart"
        fi
        if [[ $act != start ]] && [[ $act != stop ]] && [[ $act != restart ]]
        then
            echo "Wrong action: [start|stop|restart]"
            exit 0
        fi
        bgfg=$3
        if [[ x$3 == x ]]
        then
            bgfg="bg"
        fi
        if [[ x$1 == xall ]] || [[ x$1 == xai ]] || [[ x$1 == xml ]] || [[ x$1 == xcv ]] || [[ x$1 == xnlp ]] || [[ x$1 == xrl ]]
        then
            if [[ $1 == all ]]
            then
                bgfg="bg"
            fi
            $top_dir/scripts/start_services.sh dev $1 $act $bgfg
        else
            echo "Wrong task: [all|ai|ml|cv|nlp|rl]]"
        fi
    else
        echo "k12ai.sh [all|ai|ml|cv|nlp|rl] [start|stop|restart] [bg|fg]"
    fi
fi
