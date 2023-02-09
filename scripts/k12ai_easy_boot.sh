#!/bin/bash
#=================================================================
# date: 2023-02-09 18:56:38
# title: k12ai_easy_boot
# author: QRS
#=================================================================

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

images=('consul'
    'hzcsai_com/k12ai'
    'hzcsai_com/k12cv'
    'hzcsai_com/k12pyr'
)
os=ubuntu

__is_ubuntu_os()
{
    [[ -n $(awk -F= '/^NAME/{print $2}' /etc/os-release | grep -i ubuntu) ]] && echo "1" || echo "0"
}

__is_exist_img()
{
    [[ -n $(docker images --format "{{.Repository}}" $1) ]] && echo "1" || echo "0"
}

__is_exist_bin()
{
    [[ -n $(which $1) ]] && echo "1" || echo "0"
}

__apt_install()
{
    if [[ $(__is_exist_bin $1) == "0" ]]
    then
        if [[ $os == ubuntu ]]
        then
            apt install -y -f $1
        else
            yum install -y $1
        fi
    fi
}

__img_install()
{
    if [[ $(__is_exist_img $1) == "0" ]]
    then
        echo "docker pull $img"
        # docker pull $img
    fi
}

__install_softwares()
{
    __apt_install git

    ret=`__is_exist_bin docker`
    if [[ $ret == "0" ]]
    then
        __apt_install nvidia-docker2
        pkill -SIGHUP dockerd
        gpasswd -a $USER docker; newgrp docker
        cat > /etc/docker/daemon.json <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "exec-opts": ["native.cgroupdriver=systemd"],
    "log-opts": {
        "max-size": "20m",
        "max-file": "10"
    },
    "storage-driver": "overlay2",
    "data-root": "/data/docker",
    "registry-mirrors": [
        "https://p3mki545.mirror.aliyuncs.com",
        "https://registry.docker-cn.com"
    ],
    "insecure-registries": ["0.0.0.0/0"]
}
EOF
        systemctl restart docker.service
    fi
}

__start_k12ai()
{
    docker run -d --runtime nvidia --name k12ai --network host \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --env PRI_HTTP_PROXY=${HTTP_PROXY:-''} \
        --volume /var/run/docker.sock:/var/run/docker.sock \
        --volume /data:/data \
        --volume ${TOP_DIR}:${TOP_DIR} \
        --entrypoint ${TOP_DIR}/entrypoint.sh hzcsai_com/k12ai 
}


__main()
{
    if [[ $(__is_ubuntu_os) == "0" ]]
    then
        os=centos
    fi

    __install_softwares
   
    for img in ${images[@]}
    do
        __img_install $img
    done

    __start_k12ai
}

__main
