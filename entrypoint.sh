#!/bin/bash
#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

servers=("k12ai" "k12cv" "k12pyr")
ifnames=("eth0" "ens3")

__find_lanip()
{
    for ifname in ${ifnames[*]}
    do
        result=`ifconfig $ifname 2>&1 | grep -v "error"`
        if [[ x$result != x ]]
        then
            ip=`echo "$result" | grep inet\ | awk '{print $2}'`
            echo $ip
            return
        fi
    done
    exit -1
}

local_ip=$(__find_lanip)

cd $CUR_DIR
while true
do
    sleep 10000
    ./scripts/k12ai.sh all stop
    ./scripts/k12ai.sh all start 

    while true
    do
        sleep 30
        restart=0
        for chk in ${servers[@]}
        do
            res=`curl -s http://${local_ip}:8500/v1/health/checks/${chk}`
            if [[ x$res == x ]]
            then
                restart=1
                break
            fi
        done
        if [[ x$restart == x1 ]]
        then
            break
        fi
    done
done
