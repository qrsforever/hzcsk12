#!/bin/bash
#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

servers=("k12ai" "k12cv" "k12pyr")
ifnames=("eth0" "ens3")

cd $CUR_DIR
while true
do
    ./scripts/k12ai.sh all stop
    ./scripts/k12ai.sh all start 

    while true
    do
        sleep 30
        restart=0
        for chk in ${servers[@]}
        do
            res=`ps -eo pid,args | grep "$chk" | grep -v grep`
            echo "check $chk: $res"
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
