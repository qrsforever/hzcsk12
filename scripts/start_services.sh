#!/bin/bash
#=================================================================
# date: 2019-11-21 15:33:09
# title: start_services
# author: QRS
#=================================================================

cur_fil=${BASH_SOURCE[0]}
top_dir=`cd $(dirname $cur_fil)/..; pwd`
log_fil=/data/tmp/k12ai_log.txt

hostname=`hostname`
hostaddr=10.255.20.227

# service name, address and ports
consul_name=${hostname}-consul
consul_addr=$hostaddr
consul_port=8500

k12ai_service_name=${hostname}-k12ai
k12ai_addr=$hostaddr
k12ai_port=8129

k12nlp_service_name=${hostname}-k12nlp
k12nlp_addr=$hostaddr
k12nlp_port=8149

k12platform_service_name=${hostname}-k12platform
k12platform_addr=$hostaddr
k12platform_port=8119

# args of services
k12nlp_use_image="hzcsai_com/k12nlp-dev"

export HOST_NAME=${hostname}
export HOST_ADDR=${hostaddr}

# global function
__script_logout()
{
    dt=`date +"%Y-%m-%d %H:%m:%S"`
    echo $dt: $* >> $log_fil
}

__run_command()
{
    $* >/dev/null 2>&1 &
    __script_logout "$*: $!"
}

__service_health_check()
{
    if [[ x$1 == x ]]
    then
        __script_logout "need service name"
        echo "-1"
        return
    fi
    service_name=$1
    try_count=5
    if [[ x$2 != x ]]
    then
        try_count=$2
    fi

    while (( try_count > 0 ))
    do
        result=`curl -s http://${consul_addr}:${consul_port}/v1/health/checks/${service_name}|grep Success`
        if [[ x$result != x ]]
        then
            break
        fi
        sleep 1
        (( try_count = try_count - 1 ))
    done
    if (( try_count == 0 ))
    then
        __script_logout "check $service_name health fail"
        echo "0"
    fi
    echo "1"
}

# 1. check or start consul service
__start_consule_service()
{
    consul_container=`docker container ls --filter name=${consul_name} --filter status=running -q`
    if [[ x$consul_container == x ]]
    then
        if [[ ! -d /data/consul ]]
        then
            mkdir /data/consul
        fi
        docker run -dit \
            --restart=always \
            --name=${consul_name}\
            --volume /data/:/data \
            --network host \
            --hostname ${consul_name} \
            consul agent -dev -server -bootstrap \
            -http-port=${consul_port} \
            -node=${hostname} \
            -data-dir=/data/consul \
            -datacenter=${hostname} -client='0.0.0.0' -ui
        sleep 5
        __script_logout "start consul service"
    else
        __script_logout "check consul service"
    fi
}

# 2. check or start k12ai service
__start_k12ai_service()
{
    result=$(__service_health_check ${k12ai_service_name})
    if [[ $result != 1 ]]
    then
        __run_command "nohup python3 ${top_dir}/k12ai_service.py \
            --host ${k12ai_addr} \
            --port ${k12ai_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port}"
        __script_logout "start k12ai service"
    else
        __script_logout "check k12ai service"
    fi
}

# 3. check or start k12platform service
__start_k12platform_service()
{
    result=$(__service_health_check ${k12platform_service_name})
    if [[ $result != 1 ]]
    then
        __run_command "nohup python3 ${top_dir}/platform/app/app_service.py \
            --host ${k12platform_addr} \
            --port ${k12platform_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port}"
        __script_logout "start k12platform service"
    else
        __script_logout "check k12platform service"
    fi
}

# 4. check or start k12nlp service
__start_k12nlp_service()
{
    result=$(__service_health_check ${k12nlp_service_name})
    if [[ $result != 1 ]]
    then
        image=${k12nlp_use_image}
        if [[ x$1 != x ]]
        then
            image=$1
        fi
        __run_command "nohup python3 ${top_dir}/nlp/app/app_service.py \
            --host ${k12nlp_addr} \
            --port ${k12nlp_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port} \
            --image $image"
        __script_logout "start k12nlp service"
    else
        __script_logout "check k12nlp service"
    fi
}

__main()
{
    if [[ ! -d /data/tmp ]]
    then
        mkdir -p /data/tmp
    elif [[ -f ${log_fil} ]]
    then
        mv ${log_fil} ${log_fil}_bak
    fi
    if [[ x$K12AI_BREAK == x1 ]]
    then
        __script_logout "k12ai break"
        return
    fi
    cd /data/tmp
    __start_consule_service
    __start_k12ai_service
    __start_k12platform_service
    __start_k12nlp_service
}

__main $@
