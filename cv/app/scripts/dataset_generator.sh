#!/bin/bash
#=================================================================
# date: 2020-01-13 14:11:12
# title: dataset_generator
# author: QRS
#=================================================================

cur_fil=${BASH_SOURCE[0]}
top_dir=`cd $(dirname $cur_fil)/../..; pwd`
torchcv_dir=${top_dir}/torchcv

export PYTHONPATH=$torchcv_dir

echo -n "Input dataset root(default: /data/datasets/cv): "

read data_root

if [[ x$data_root == x ]]
then
    data_root=/data/datasets/cv
fi

echo -e "\n"
echo -e "\t1. cifar10"
echo -e "\t2. mnist"
echo -e "\n"

echo -n "Select dataset:"

read select

case $select in
    '1')
        dataset='cifar10'
        save_dir=$data_root/cifar10/
        root_dir=$save_dir/cifar-10-batches-py 
        if [[ -d $root_dir ]]
        then
            cd $torchcv_dir/data/cls/preprocess/cifar
            python3 cifar_cls_generator.py --root_dir $root_dir --save_dir $save_dir --dataset $dataset
            cd - > /dev/null
        else
            echo "Not found $root_dir"
        fi
        ;;
    *)
        echo "dataset select error"
        ;;
esac
