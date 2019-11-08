#!/bin/bash
#=================================================================
# date: 2019-08-31 18:11:04
# title: start_docker
# author: QRS
#=================================================================

REPOSITORY="hzcsai_com/waltzcv"

items=($(docker images --filter "label=org.label-schema.name=$REPOSITORY" --format "{{.Repository}}:{{.Tag}}"))
count=${#items[@]}

if (( $count == 0 ))
then
    echo "not found correct image!"
    exit 0
fi

if (( $count > 1 ))
then
    i=0
    while (( i < $count ))
    do
        echo "$i. ${items[$i]}"
        (( i = i + 1 ))
    done
    echo -n "Select: "
    read select
else
    select=0
fi

echo "use image: ${items[$select]}"

container=`docker container  ls --filter ancestor=${items[$select]} --filter status=running -q`

if [[ x$container != x ]]
then
    echo "${items[$select]}: $container already start, you can stop/remove it before start"
    exit 0
fi

containers=(`docker container  ls --filter name=waltzcv --filter status=running -q`)

for c in $containers
do
    echo "stop $c"
    docker container stop $c
done

docker inspect ${items[$select]} --format  '{{json .ContainerConfig.Labels}}' | python -m json.tool

cmd=$(docker inspect ${items[$select]} --format '{{index .ContainerConfig.Labels "org.label-schema.docker.cmd"}}')

if [[ x$cmd != x ]]
then
    $cmd
else
    echo "not found command in org.label-schema.docker.cmd"
fi
