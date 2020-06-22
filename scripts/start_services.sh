#!/bin/bash
#=================================================================
# date: 2019-11-21 15:33:09
# title: start_services
# author: QRS
#=================================================================


EXE=`ps -o comm= $PPID`
if [[ $EXE != 'k12ai.sh' ]]
then
    echo "Can't run by itself, You can run k12ai.sh instead."
    exit 0
fi

cur_fil=${BASH_SOURCE[0]}
top_dir=`cd $(dirname $cur_fil)/..; pwd`

debug=1
use_image='unkown'

ifnames=("eth0" "ens3")
__find_lanip() {
    for ifname in ${ifnames[*]}
    do
        result=`ifconfig $ifname 2>&1 | grep -v "error"`
        if [[ x$result != x ]]
        then
            ip=`echo "$result" | grep inet\ | awk '{print $2}' | awk -F: '{print $2}'`
            echo $ip
            return
        fi
    done
    exit -1
}

__find_netip() {
    result=`curl -s icanhazip.com`
    if [[ x$result != x ]]
    then
        echo "$result"
        return
    fi
    result=`wget -qO - ifconfig.co`
    if [[ x$result != x ]]
    then
        echo "$result"
        return
    fi
    result=`curl ipecho.net/plain`
    if [[ x$result != x ]]
    then
        echo "$result"
        return
    fi
    exit -1
}

hostname=`hostname`
hostlanip=$(__find_lanip)
hostnetip=$(__find_netip)

echo -e "\n####\thostname($hostname), hostlanip($hostlanip), hostnetip($hostnetip)\t####"

if [[ x$hostnetip == x ]]
then
    echo "Cann't get netip"
    exit -1
fi

log_fil=/tmp/k12ai_log.txt
k12logs=/tmp/k12logs

# service name, address and ports
redis_addr=${REDIS_ADDR:-'116.85.56.62'}
redis_port=10090
redis_pswd='qY3Zh4xLPZNMkaz3'

minio_server_url='s3-internal.didiyunapi.com'
minio_access_key='AKDD002E38WR1J7RMPTGRIGNVCVINY'
minio_secret_key='ASDDXYWs45ov7MNJbj5Wc2PM9gC0FSqCIkiyQkVC'

consul_name=consul-dev
consul_addr=$hostlanip
consul_port=8500

k12ai_service_name=k12ai
k12ai_addr=$hostlanip
k12ai_port=8119

k12ml_service_name=k12ml
k12ml_addr=$hostlanip
k12ml_port=8129

k12cv_service_name=k12cv
k12cv_addr=$hostlanip
k12cv_port=8139

k12nlp_service_name=k12nlp
k12nlp_addr=$hostlanip
k12nlp_port=8149

k12rl_service_name=k12rl
k12rl_addr=$hostlanip
k12rl_port=8159

k123d_service_name=k123d
k123d_addr=$hostlanip
k123d_port=8169

dataset_port=9090

export HOST_NAME=${hostname}
export HOST_LANIP=${hostlanip}
export HOST_NETIP=${hostnetip}
export PYTHONPATH=${top_dir}/common:$PYTHONPATH

export MINIO_SERVER_URL=$minio_server_url
export MINIO_ACCESS_KEY=$minio_access_key
export MINIO_SECRET_KEY=$minio_secret_key

# global function
__script_logout()
{
    dt=`date +"%Y-%m-%d %H:%M:%S"`
    echo $dt: $* | tee -a $log_fil
}

__run_command()
{
    if [[ x$1 != xnohup ]]
    then
        $*
        return
    fi
    $* >/dev/null 2>&1 &
    __script_logout "$*: $!"
}


__kill_service()
{
    if [[ x$1 == xall ]]
    then
        IFS_OLD=$IFS
        IFS=$'\n'
        for str in `ps -eo pid,args | grep k12.*_service | grep -v grep`
        do
            if [[ x$str != x ]]
            then
                pid=`echo $str | awk '{print $1}'`
                if [[ x$pid != x ]]
                then
                    kill -9 $pid
                    __script_logout "(killed) $str"
                fi
            fi
        done
        IFS=$IFS_OLD
    elif [[ x$1 != x ]]
    then
        str=`ps -eo pid,args | grep $1_service | grep -v grep`
        if [[ x$str != x ]]
        then
            pid=`echo $str | awk '{print $1}'`
            if [[ x$pid != x ]]
            then
                kill -9 $pid
                __script_logout "(killed) $str"
            fi
        fi
    fi
}

__software_install_check()
{
    ret=`dpkg-query -W --showformat='${Package}' $1 | grep "no packages"`
    if [[ x$ret == x ]]
    then
        echo "1"
    else
        echo "0"
    fi
}

__service_environment_check()
{
    if [[ ! -d /data/ ]]
    then
        sudo mkdir /data
        sudo chmod 777 /data
        echo "##############"
        echo "No /data dir, should create the dir or mount remote dir"
        echo "##############"
        exit -1
    fi

    if [[ ! -d /data/nltk_data ]]
    then
        echo "##############"
        echo "Warning: not found nltk_data, downloading from https://github.com/nltk/nltk_data"
        echo "##############"
    fi

    ret=$(__software_install_check nfs-kernel-server)
    if [[ $ret == "0" ]]
    then
        echo "##############"
        echo "1. sudo apt install -y nfs-kernel-server"
        echo "2. if server: echo '/data 10.xx.xx.*(rw,sync,no_root_squash,no_subtree_check)' >> /etc/exports"
        echo "3. if server: sudo /etc/init.d/nfs-kernel-server restart"
        echo "##############"
        exit -1
    fi

    ret=$(__software_install_check nfs-common)
    if [[ $ret == "0" ]]
    then
        echo "##############"
        echo "1. sudo apt install -y nfs-common"
        echo "2. add dataserver host, eg: "10.255.0.229 dataserver" in /etc/hosts"
        echo "3. if client: sudo mount -t nfs server_ip:/data /data"
        echo "Tips: ip must be lan address, like 10.xxx.xxx.xx"
        echo "##############"
        exit -1
    fi

    ret=$(__software_install_check docker)
    if [[ $ret == "0" ]]
    then
        echo "##############"
        echo "Please install docker and nvidia-docker manually, then run deps_install.sh"
        echo "Edit/create the /etc/docker/daemon.json"
        echo "{"
        echo "    \"runtimes\": {"
        echo "       \"nvidia\": {"
        echo "            \"path\": \"nvidia-container-runtime\","
        echo "            \"runtimeArgs\": []",
        echo "        }"
        echo "    },"
        echo "    \"default-runtime\": \"nvidia\""
        echo "}"
        echo ""
        echo "sudo apt-get install nvidia-container-runtime"
        echo "sudo systemctl restart docker.service"
        echo "##############"
        exit -1
    fi
}

__service_health_check()
{
    if [[ x$1 == x ]]
    then
        # __script_logout "need service name"
        echo "-1"
        return
    fi
    service_name=$1

    str=`ps -eo pid,args | grep ${service_name}_service | grep -v grep`
    if [[ x$str == x ]]
    then
        echo "0"
        return
    fi

    try_count=3
    if [[ x$2 != x ]]
    then
        try_count=$2
    fi

    while (( try_count > 0 ))
    do
        result=`curl -s http://${consul_addr}:${consul_port}/v1/health/checks/${service_name} | grep Success`
        if [[ x$result != x ]]
        then
            break
        fi
        sleep 1
        (( try_count = try_count - 1 ))
    done
    if (( try_count == 0 ))
    then
        # __script_logout "check $service_name health fail"
        echo "0"
    fi
    echo "1"
}

__service_image_check()
{
    REPOSITORY=$1
    items=($(docker images --filter "label=org.label-schema.name=$REPOSITORY" --format "{{.Tag}}"))
    count=${#items[@]}

    if (( $count == 0 ))
    then
        __script_logout "No $REPOSITORY image: exc build_base.sh and all start_docker.sh in subdir"
        exit -1
    else
        i=0
        while (( i < $count ))
        do
            echo "$i. $REPOSITORY:${items[$i]}"
            (( i = i + 1 ))
        done
        if (( $i > 1 ))
        then
            echo -n "Select: "
            read select
            use_image=$REPOSITORY:${items[$select]}
        else
            use_image=$REPOSITORY:${items[0]}
        fi
    fi
}

# 1. check or start consul service
__start_consul_service()
{
    consul_container=`docker container ls --filter name=${consul_name} --filter status=running -q`
    if [[ x$consul_container == x ]]
    then
        if [[ ! -d /srv/consul ]]
        then
            sudo mkdir /srv/consul
            sudo chmod 777 /srv/consul
        fi
        docker run -dit \
            --restart=always \
            --name=${consul_name}\
            --volume /srv/consul:/srv/consul \
            --network host \
            --hostname ${consul_name} \
            consul agent -dev -server -bootstrap \
            -http-port=${consul_port} \
            -node=${hostname} \
            -data-dir=/srv/consul \
            -datacenter=k12ai -client='0.0.0.0' -ui
        __script_logout "start consul service"
    else
        __script_logout "consul service is ok."
    fi
}

# 2. check or start k12ai service
__start_k12ai_service()
{
    [ x$1 != xstart ] && __kill_service ${k12ai_service_name}
    if [[ x$1 == xstop ]]
    then
        return
    fi
    if [[ x$2 == xfg ]]
    then
        cmdstr=
    else
        cmdstr="nohup"
    fi
    result=$(__service_health_check ${k12ai_service_name})
    if [[ $result != 1 ]]
    then
        export K12AI_DEBUG=$debug
        cmdstr="$cmdstr python3 ${top_dir}/services/k12ai_service.py \
            --host ${k12ai_addr} \
            --port ${k12ai_port} \
            --redis_addr ${redis_addr} \
            --redis_port ${redis_port} \
            --redis_passwd ${redis_pswd} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port}"

        __run_command $cmdstr
        __script_logout "start k12ai service"
    else
        __script_logout "k12ai service is already running"
    fi
}

# 3. check or start k12ml service
__start_k12ml_service()
{
    [ x$1 != xstart ] && __kill_service ${k12ml_service_name}
    if [[ x$1 == xstop ]]
    then
        return
    fi
    use_image="hzcsai_com/k12ml"
    if [[ x$2 == xfg ]]
    then
        result=0
        cmdstr=
    else
        result=$(__service_health_check ${k12ml_service_name})
        cmdstr="nohup"
    fi
    if [[ $result != 1 ]]
    then
        export K12ML_DEBUG=$debug
        cmdstr="$cmdstr python3 ${top_dir}/services/k12ml_service.py \
            --host ${k12ml_addr} \
            --port ${k12ml_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port} \
            --image $use_image"

        __run_command $cmdstr
        __script_logout "start k12ml service"
    else
        __script_logout "k12ml service is already running"
    fi
}

# 4. check or start k12cv service
__start_k12cv_service()
{
    [ x$1 != xstart ] && __kill_service ${k12cv_service_name}
    if [[ x$1 == xstop ]]
    then
        return
    fi
    use_image="hzcsai_com/k12cv"
    if [[ x$2 == xfg ]]
    then
        result=0
        cmdstr=
    else
        result=$(__service_health_check ${k12cv_service_name})
        cmdstr="nohup"
    fi
    if [[ $result != 1 ]]
    then
        export K12CV_DEBUG=$debug
        cmdstr="$cmdstr python3 ${top_dir}/services/k12cv_service.py \
            --host ${k12cv_addr} \
            --port ${k12cv_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port} \
            --image $use_image"

        __run_command $cmdstr
        __script_logout "start k12cv service"
    else
        __script_logout "k12cv service is already running"
    fi
}

# 5. check or start k12nlp service
__start_k12nlp_service()
{
    [ x$1 != xstart ] && __kill_service ${k12nlp_service_name}
    if [[ x$1 == xstop ]]
    then
        return
    fi
    use_image="hzcsai_com/k12nlp"
    if [[ x$2 == xfg ]]
    then
        result=0
        cmdstr=
    else
        result=$(__service_health_check ${k12nlp_service_name})
        cmdstr="nohup"
    fi
    if [[ $result != 1 ]]
    then
        export K12NLP_DEBUG=$debug
        cmdstr="$cmdstr python3 ${top_dir}/services/k12nlp_service.py \
            --host ${k12nlp_addr} \
            --port ${k12nlp_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port} \
            --image $use_image"

         __run_command $cmdstr
        __script_logout "start k12nlp service"
    else
        __script_logout "k12nlp service is already running"
    fi
}

# 6. check or start k12rl service
__start_k12rl_service()
{
    [ x$1 != xstart ] && __kill_service ${k12rl_service_name}
    if [[ x$1 == xstop ]]
    then
        return
    fi
    use_image="hzcsai_com/k12rl"
    if [[ x$2 == xfg ]]
    then
        result=0
        cmdstr=
    else
        result=$(__service_health_check ${k12rl_service_name})
        cmdstr="nohup"
    fi
    if [[ $result != 1 ]]
    then
        export K12RL_DEBUG=$debug
        cmdstr="$cmdstr python3 ${top_dir}/services/k12rl_service.py \
            --host ${k12rl_addr} \
            --port ${k12rl_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port} \
            --image $use_image"

        __run_command $cmdstr
        __script_logout "start k12rl service"
    else
        __script_logout "k12rl service is already running"
    fi
}

# 7. check or start k123d service
__start_k123d_service()
{
    [ x$1 != xstart ] && __kill_service ${k123d_service_name}
    if [[ x$1 == xstop ]]
    then
        return
    fi
    use_image="hzcsai_com/k123d"
    if [[ x$2 == xfg ]]
    then
        result=0
        cmdstr=
    else
        result=$(__service_health_check ${k123d_service_name})
        cmdstr="nohup"
    fi
    if [[ $result != 1 ]]
    then
        export K12CV_DEBUG=$debug
        cmdstr="$cmdstr python3 ${top_dir}/services/k123d_service.py \
            --host ${k123d_addr} \
            --port ${k123d_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port} \
            --image $use_image"

        __run_command $cmdstr
        __script_logout "start k123d service"
    else
        __script_logout "k123d service is already running"
    fi
}

# 8. setart dataset service
__start_dataset_service()
{
    result=`ps -eo pid,args | grep "http.server $dataset_port" | grep -v grep`
    if [[ x$result == x ]]
    then
        cd $1
        __run_command nohup python3 -m http.server $dataset_port
        cd - > /dev/null
    else
        __script_logout "http.server $dataset_port already run"
    fi
}

__main()
{
    if [[ -f ${log_fil} ]]
    then
        sudo mv ${log_fil} ${log_fil}_bak
    else
        sudo touch ${log_fil}
        sudo chmod 777 ${log_fil}
    fi
    if [[ ! -d $k12logs ]]
    then
        mkdir -p $k12logs
    fi

    if [[ x$1 == xrelease ]]
    then
        debug=0
    fi

    __service_environment_check

    cd $k12logs
    __start_consul_service
    [ $2 == all -o $2 == ai ]  && __start_k12ai_service  $3 $4 
    [ $2 == all -o $2 == ml ]  && __start_k12ml_service  $3 $4
    [ $2 == all -o $2 == cv ]  && __start_k12cv_service  $3 $4
    [ $2 == all -o $2 == rl ]  && __start_k12rl_service  $3 $4
    [ $2 == all -o $2 == nlp ] && __start_k12nlp_service $3 $4
    [ $2 == all -o $2 == 3d ]  && __start_k123d_service  $3 $4
    cd - > /dev/null

    __start_dataset_service /data
}

__main $@
