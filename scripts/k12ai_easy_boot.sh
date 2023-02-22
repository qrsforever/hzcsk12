#!/bin/bash
#=================================================================
# date: 2023-02-09 18:56:38
# title: k12ai_easy_boot
# author: QRS
#=================================================================

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`
DOCKER_HOST=hzcsk8s.io

images=(
    "hzcsai_com/k12ai"
    "hzcsai_com/k12cv"
    "hzcsai_com/k12pyr"
)

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

# __is_ubuntu_os()
# {
#     [[ -n $(awk -F= '/^NAME/{print $2}' /etc/os-release | grep -i ubuntu) ]] && echo "1" || echo "0"
# }

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
        if [[ $distribution =~ "ubuntu" ]]
        then
            apt install -y -f $1
        else
            dnf install -y $1
        fi
    fi
}

__img_install()
{
    [[ $(__is_exist_img consul) == "0" ]] && docker pull consul

    for img in ${images[@]}
    do
        if [[ $(__is_exist_img $img) == "0" ]]
        then
            echo "docker pull ${DOCKER_HOST}/$img"
            docker pull ${DOCKER_HOST}/$img
            docker tag ${DOCKER_HOST}/$img $img
        fi
    done
}

__install_softwares()
{
    __apt_install git
    [[ $distribution =~ "centos" ]] && __apt_install mount.nfs
    # locate error:  updatedb
    # [[ $distribution =~ "centos" ]] && updatedb
        
    ret=`__is_exist_bin docker`
    if [[ $ret == "0" ]]
    then
        # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
        if [[ $distribution =~ "centos8" ]]
        then
            dnf install -y tar bzip2 make automake gcc gcc-c++ vim pciutils elfutils-libelf-devel libglvnd-devel iptables
            dnf config-manager --add-repo=https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
            dnf repolist -v
            dnf install -y https://mirrors.aliyun.com/docker-ce/linux/centos/8/x86_64/stable/Packages/containerd.io-1.4.3-3.1.el8.x86_64.rpm --allowerasing
            dnf install -y docker-ce
            curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
            dnf clean expire-cache --refresh
            dnf install -y nvidia-container-toolkit
            nvidia-ctk runtime configure --runtime=docker
        else
            __apt_install nvidia-docker2
        fi
        pkill -SIGHUP dockerd
        gpasswd -a $USER docker # ; newgrp docker
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
        systemctl enable docker
        systemctl restart docker
    fi
}

__start_k12ai()
{
    # ubuntu: /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
    # centos: ?
    nvidia_so_path=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
    if [[ $distribution =~ "centos" ]]
    then
        nvidia_so_path=xxx
    fi
    docker run -d --runtime nvidia --name k12ai --network host --pid host --privileged \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --env PRI_HTTP_PROXY=${HTTP_PROXY:-''} \
        --volume /var/run/docker.sock:/var/run/docker.sock \
        --volume /data:/data \
        --volume /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
        --volume ${nvidia_so_path}:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
        --volume /tmp/k12logs:/tmp/k12logs \
        --volume ${TOP_DIR}:${TOP_DIR} \
        --entrypoint ${TOP_DIR}/entrypoint.sh hzcsai_com/k12ai 
}


__main()
{
    __install_softwares
   
    __img_install

    __start_k12ai
}

__main
