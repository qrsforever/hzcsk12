#!/bin/bash
#=================================================================
# date: 2019-11-21 15:33:09
# title: start_services
# author: QRS
#=================================================================

cur_fil=${BASH_SOURCE[0]}
top_dir=`cd $(dirname $cur_fil)/..; pwd`

debug=0

hostname=`hostname`
hostaddr=`ifconfig eth0| grep inet\ | awk '{print $2}' | awk -F: '{print $2}'`

log_fil=/tmp/k12ai_log.txt

# service name, address and ports
consul_name=${hostname}-consul
consul_addr=$hostaddr
consul_port=8500

k12ai_service_name=${hostname}-k12ai
k12ai_addr=$hostaddr
k12ai_port=8119

k12platform_service_name=${hostname}-k12platform
k12platform_addr=$hostaddr
k12platform_port=8129

k12cv_service_name=${hostname}-k12cv
k12cv_addr=$hostaddr
k12cv_port=8139

k12nlp_service_name=${hostname}-k12nlp
k12nlp_addr=$hostaddr
k12nlp_port=8149

export HOST_NAME=${hostname}
export HOST_ADDR=${hostaddr}

# global function
__script_logout()
{
    dt=`date +"%Y-%m-%d %H:%m:%S"`
    echo $dt: $* | tee -a $log_fil
}

__run_command()
{
    $* >/dev/null 2>&1 &
    __script_logout "$*: $!"
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
        echo "2. if client: sudo mount -t nfs server_ip:/data /data"
        echo "##############"
        exit -1
    fi

    ret=$(__software_install_check docker)
    if [[ $ret == "0" ]]
    then
        echo "##############"
        echo "Please install docker and nvidia-docker manually, then run deps_install.sh"
        echo "##############"
        exit -1
    fi
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
        __script_logout "check $service_name health fail"
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
__start_consule_service()
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
            -datacenter=${hostname} -client='0.0.0.0' -ui
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
        __run_command "nohup python3 ${top_dir}/platform/app/k12platform_service.py \
            --host ${k12platform_addr} \
            --port ${k12platform_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port}"
        __script_logout "start k12platform service"
    else
        __script_logout "check k12platform service"
    fi
}

# 4. check or start k12cv service
__start_k12cv_service()
{
    result=$(__service_health_check ${k12cv_service_name})
    if [[ $result != 1 ]]
    then
        __service_image_check "hzcsai_com/k12cv"
        __run_command "K12CV_DEBUG=$debug nohup python3 ${top_dir}/cv/app/k12cv_service.py \
            --host ${k12cv_addr} \
            --port ${k12cv_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port} \
            --image $use_image"
        __script_logout "start k12cv service"
    else
        __script_logout "check k12cv service"
    fi
}

# 5. check or start k12nlp service
__start_k12nlp_service()
{
    result=$(__service_health_check ${k12nlp_service_name})
    if [[ $result != 1 ]]
    then
        __service_image_check "hzcsai_com/k12nlp"
        __run_command "K12NLP_DEBUG=$debug nohup python3 ${top_dir}/nlp/app/k12nlp_service.py \
            --host ${k12nlp_addr} \
            --port ${k12nlp_port} \
            --consul_addr ${consul_addr} \
            --consul_port ${consul_port} \
            --image $use_image"
        __script_logout "start k12nlp service"
    else
        __script_logout "check k12nlp service"
    fi
}

__main()
{
    if [[ x$1 == xdev ]]
    then
        debug=1
    fi
    __service_environment_check

    if [[ -f ${log_fil} ]]
    then
        mv ${log_fil} ${log_fil}_bak
    fi
    cd /tmp
    __start_consule_service
    __start_k12ai_service
    __start_k12platform_service
    __start_k12cv_service
    __start_k12nlp_service
    cd - > /dev/null
}

__main $@
