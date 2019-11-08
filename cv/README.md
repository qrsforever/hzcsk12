---

title: 框架运行docker设计

date: 2019-08-31 13:53:32
tags: [Cauchy]
categories: [Company]

---

<!-- vim-markdown-toc GFM -->

* [如何启动框架](#如何启动框架)
* [三层镜像](#三层镜像)
* [制作镜像](#制作镜像)
    * [制作第二层](#制作第二层)
    * [制作第三层](#制作第三层)
* [简单总结](#简单总结)

<!-- vim-markdown-toc -->

<!-- more -->

# 如何启动框架

框架的部署使用docker, docker依赖镜像, 从缩短部署时间和减少带宽浪费的角度设计, 我们的框架使用三层镜像, 未来
框架足够稳定, 可以做一个完整的镜像"pull"镜像服务器上, 先介绍三层镜像构建和启动.

在介绍之前先把常用的docker命令做个Alias, 后面一些例子会用到.

```bash
alias di='docker images'
alias dri='docker rmi' # $(docker images -q)'

alias dc='docker container ls -a'
alias drc='docker container rm'
alias dsc='docker container stop'

alias dv='docker volume ls'
alias drv='docker volume rm'

alias dip='docker inspect --format="\{\{range .NetworkSettings.Networks\}\}\{\{.IPAddress\}\}\{\{end\}\}"'
alias dit='docker run -it'

alias din='docker inspect'
alias dlg='docker logs --follow'

dsh()
{
    args=($@)
    container=$1
    bashcmd=${args[@]: 1:$# }
    docker exec $container bash -c "$bashcmd"
}
```

# 三层镜像

cmd: `colorai@10-255-0-185:~$ di`

output:

```
   第三层: 咱们框架自身,排除三方库, 可以认为就是框架代码自身安装, 可能更新会频繁.
                             ^
                             |
                             |
REPOSITORY                   |  TAG                 IMAGE ID            CREATED             SIZE
colorai/cauchycv_visdom |    |  0.2.7               5d3ab89af6d4        15 hours ago        6.3GB
colorai/cauchycv_visdom |----+  0.2.6               98a757ed7816        16 hours ago        6.3GB
colorai/cauchycv_visdom |       0.2.5               7148c78368fb        20 hours ago        6.3GB
colorai/cauchycv_base           latest              83c5742fc484        22 hours ago        6.06GB
ufoym/deepo    |                pytorch-py36-cu90   242fdb23e0b0        5 months ago        4.53GB
    |          |
    |          +----->  第二层: 框架运行时依赖的第三方软件和必要工具, 基于第一层又加了不少, 更新不频繁
    |
    |
    +---> 第一层: 包含机器学习框架pytorch需要必要软件工具, 几乎不更新

```


有了第二层(本地镜像)的出现, 在更新第三层时, 就会少很多三方软件的安装, 节省时间和带宽.

# 制作镜像

先介绍一下, 版本信息的设计和存储方案, 为以后定位问题提供帮助.

[CauchyCV框架代码](https://gitee.com/colorai/sig_cauchy_cv)使用git管理, 所以很多信息是从git命令中获取.

例如: `build_docker.sh`中的部分代码:

```bash

DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=$(git describe --tags --always)
URL=$(git config --get remote.origin.url)
COMMIT=$(git rev-parse HEAD | cut -c 1-7)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

TAG="0.2.$(git rev-list HEAD | wc -l | awk '{print $1}')"

```

分别提取出: 当前时间, 版本, 代码URL, 最后一次提交commit-id, 以及分支号, 还有TAG作为镜像的版本, 如`0.2.7`,
其中后面的7是代码所有提交的总个数, 是一个递增的数字. 所有这些变量会在build docker时传入到镜像的labels中.


## 制作第二层

比较简单, 基于第一层**ufoym/deepo**, 然后`apt install` 和 `pip install`安装框架需要的软件, 同时也安装咱们自
己维护的[外部库](https://gitee.com/colorai/sig_cauchy_external), 这些外部库因为github上更新较为频繁, 对我
们调试开发会带来不可知的影响, 所以单独管理起来, 其中还包括`vulcan`.

提供了一个`DockerFile.base`文件, 构建第二层镜像直接指定这个文件即可, 不要忘了外部库要提前下载.

例如: `build_docker.sh`中的部分代码:

```sh
base_image=colorai/cauchycv_base

if [ ! -d external ]
then
    git clone git@gitee.com:colorai/sig_cauchy_external.git external
fi

base_tag=`docker images -q $base_image:latest`

if [[ x$base_tag == x ]]
then
     echo "build $base_image"
     docker build --tag $base_image:latest \
                  --build-arg DATE=$DATE \
                  --file Dockerfile.base .
fi
```

值得注意的是Base镜像指定的TAG版本固定为"latest", 即本地只会(或者要保证)有一个.

详细信息可查看[DockerFile.base](https://gitee.com/colorai/sig_cauchy_cv/blob/master/Dockerfile.base)

## 制作第三层

框架层镜像一个重点就是描述好本镜像自身, 然后安装框架到镜像中即可, 如何把上面提到的版本信息保存在镜像中?

Docker提供LABEL命令, 可以实现该功能, 先看看下面的实例(Dockerfile部分代码):

```shell
LABEL maintainer="colorai@colorai.com"

ARG VENDOR="ColorAI"
ARG REPOSITORY
ARG TAG
ARG DATE
ARG VERSION
ARG URL
ARG COMMIT
ARG BRANCH

LABEL org.label-schema.schema-version="1.0" \
      org.label-schema.build-date=$DATE \
      org.label-schema.name=$REPOSITORY \
      org.label-schema.description="Computer Vision Backend for Cauchy" \
      org.label-schema.url=https://www.colorai.com/index.php?r=front \
      org.label-schema.vcs-url=$URL \
      org.label-schema.vcs-ref=$COMMIT \
      org.label-schema.vcs-branch=$BRANCH \
      org.label-schema.vendor=$VENDOR \
      org.label-schema.version=$VERSION \
      org.label-schema.docker.cmd="docker run -d --name framework \
--restart unless-stopped --volume /data:/data --network host --hostname colorai \
--runtime nvidia --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
$REPOSITORY:$TAG python cauchy_services.py --port 8339"
```

ARG这一堆的变量是在build镜像时传递过来的,  `build_docker.sh`中的部分代码:

```shell
docker build --tag $REPOSITORY:$TAG \
             --build-arg REPOSITORY=$REPOSITORY \
             --build-arg TAG=$TAG \
             --build-arg DATE=$DATE \
             --build-arg VERSION=$VERSION \
             --build-arg URL=$URL \
             --build-arg COMMIT=$COMMIT \
             --build-arg BRANCH=$BRANCH \
             .
```

需要对**org.label-schema.docker.cmd**这个label说明一下, 镜像建立好之后, 需要在该镜像的Container上运行咱们
的框架, 如何运行呢, 不同的框架版本是不是运行的命令和参数不一样, 如果框架版本越来越多, 我如何快速的知道如何
运行指定版本的框架(执行的命令和参数是什么), 当然我们可以通过**org.label-schema.version**和
**org.label-schema.vcs-ref**确定如何执行该container. 为了更简单快捷, 加了一个**org.label-schema.docker.cmd**
参数,它的值表示为**推荐**执行该container的命令, 这里用**推荐**二字, 是因为不是必须的.

建立好镜像后, 如何查看:

command: `din 5d3ab89af6d4 --format  '\{\{json .ContainerConfig.Labels\}\}' | python -m json.tool`

output:

```json
{
    "com.nvidia.cuda.version": "9.0.176",
    "com.nvidia.cudnn.version": "7.4.2.24",
    "com.nvidia.volumes.needed": "nvidia_driver",
    "maintainer": "colorai@colorai.com",
    "org.label-schema.build-date": "2019-08-30T14:46:20Z",
    "org.label-schema.description": "Computer Vision Backend for Cauchy",
    "org.label-schema.docker.cmd": "docker run -d --name cauchy_service --restart unless-stopped --volume /data:/data --network host --hostname colorai --runtime nvidia --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 colorai/cauchycv_visdom:0.2.7 python cauchy_services.py --port 8339",
    "org.label-schema.name": "colorai/cauchycv_visdom",
    "org.label-schema.schema-version": "1.0",
    "org.label-schema.url": "https://www.colorai.com/index.php?r=front",
    "org.label-schema.vcs-branch": "master",
    "org.label-schema.vcs-ref": "bbe9a6a",
    "org.label-schema.vcs-url": "git@gitee.com:colorai/sig_cauchy_cv.git",
    "org.label-schema.vendor": "ColorAI",
    "org.label-schema.version": "bbe9a6a"
}
```

另外提供了一个**start_docker.sh**脚本用来启动container, 脚本中我会提取出**org.label-schema.docker.cmd**直
接执行.

例如: `start_docker.sh`中的部分代码:

```sh
cmd=$(docker inspect ${items[$select]} --format '\{\{index .ContainerConfig.Labels "org.label-schema.docker.cmd"\}\}')

if [[ x$cmd != x ]]
then
    $cmd
else
    echo "not found command in org.label-schema.docker.cmd"
fi
```

详细信息可查看[DockerFile](https://gitee.com/colorai/sig_cauchy_cv/blob/master/Dockerfile)

# 简单总结

- 创建镜像[`build_docker.sh`](https://gitee.com/colorai/sig_cauchy_cv/blob/master/build_docker.sh)

    先判断是否已经创建了`cauchycv_base`(第二层镜像), 如果已经创建了, 直接创建`colorai/cauchycv_xxx`(第三
    层镜像).

- 运行镜像[`start_docker.sh`](https://gitee.com/colorai/sig_cauchy_cv/blob/master/start_docker.sh)

    先获取docker中有多少个Cauchy框架镜像, 如果有多个, 提示选择使用哪个版本, 然后判断该版本的cauchy服务是否
    已经启动, 如果已经启动则直接退出, 接着判断是否已经启动了其他版本的cauchy服务, 如果启动了, 停止掉他们,
    因为启动的服务现在使用的network为host, 已启动的cauchy服务器已经把端口暂用了, 最后从对应版本的label中获
    取到**推荐**用的**org.label-schema.docker.cmd**执行服务器的命令, 直接执行.

# 补充

启动cauchy服务(container)命令参数的介绍

```
docker run -d \                              # d: 后台运行
      --name cauchy_service \                # container 名字, 可用于display和filter
      --restart unless-stopped \             # 重启机制: unless-stopped, 只要不是手动停止, 无论服务怎么退出的, 都会重新启动, 即使host系统reboot
      --volume /data:/data \                 # 共享文件, 将host机的/data/映射到container中/data目录
      --network host --hostname colorai \    # 采用host方式的网络, 即使用host机的网络栈, 主机名为colorai
      --runtime nvidia \                     # 运行时状态为GPU容器, GPU程序可以执行
      --shm-size=2g \                        # 共享内存2GB
      --ulimit memlock=-1 \                  # 进程资源限制: 最大锁定内存地址空间，-1表示不限制
      --ulimit stack=67108864 \              # 进程资源限制: 栈最大64MB
      colorai/cauchycv_visdom:0.2.7 \        # 容器运行在哪个镜像里
      python cauchy_services.py --port 8339  # 在容器里执行的程序
```

共享目录/data,这也是重启容器时不会影响到的目录:

```
/data
  ├── aicron.sh          # 定时check任务脚本, 由/etc/crontab驱动, 目前只有启动"预置权重http服务"
  ├── checkpoints        # 客户训练模型时checkpoints目录
  ├── datasets           # 常用的数据集
  ├── pretrained_models  # Internet上已经训练好的模型参数
  └── projects           # 客户训练模型时产生的中间文件,如log
```
