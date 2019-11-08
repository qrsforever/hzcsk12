#!/bin/bash

# uninstall cauchy
pip uninstall -y cauchy

# update package
cd ../../
python setup.py install

# change dir
cd test/cauchy

VIZ_PORT=8097
# test classification
DATA_DIR="/home/dc2-user/online/data/cifar10"
MODEL_NAME="shufflenetv2"
LOSS_TYPE="fc_ce_loss"
CHECKPOINTS_NAME="fc_${MODEL_NAME}_cifar10_cls"
# PRETRAINED_MODEL="~/data/pretrained_models/vgg19-dcbb9e9d.pth"
MAX_EPOCH=400


# testing classification train phase
python main.py --hypes hypes/cls/cifar/fc_vgg19_cifar10_cls.json \
               --phase train \
               --model_name ${MODEL_NAME} \
               --gpu 0 \
               --gathered n \
               --viz_port ${VIZ_PORT}

# testing classification test phase
# python main.py --hypes hypes/cls/cifar/fc_vgg19_cifar10_cls.json \
#                --phase test \
#                --model_name ${MODEL_NAME} \
#                --gpu 0 \
#                --resume ./proj1/checkpoints/cls/cifar10/run1/fc_vgg19_cifar10_cls_max_performance.pth \
#                --test_dir ${DATA_DIR}  

# test Detection
# python main.py --hypes hypes/det/voc/yolov3_darknet_voc_det.json \
#               --phase train \
#               --model_name darknet_yolov3 \
#               --log_to_file y \
#               --gpu 0 \
#               --viz_port ${VIZ_PORT}


# DATA_DIR="/home/hancock/data/ade20k_pre"

# # PRETRAINED_MODEL="/home/hancock/data/pretrained_models/resnet50-19c8e357.pth"

# HYPES_FILE='hypes/seg/ade20k/fs_pspnet_ade20k_seg.json'
# # HYPES_FILE='hypes/seg/ade20k/seg_test.json'
# DATA_DIR='/home/hancock/data/ade20k'
# MAX_ITERS=40000

# python main.py --hypes ${HYPES_FILE} --drop_last y --phase train --gathered n --loss_balance y \
#                --gpu 0 --log_to_file y \
#                --data_dir ${DATA_DIR} --max_iters ${MAX_ITERS}\
#                --viz_port 8097 


# # testing classification train phase
# search network
# python main.py --hypes hypes/automl/darts.json \
#                --phase train \
#                --gpu 0 \
#               T --viz_port ${VIZ_PORT}
# finetune network network
# python main.py --hypes hypes/automl/darts_finetune.json --drop_last y \
#                --phase finetune  --gathered n --log_to_file y \
#                --gpu 0 \
#                --viz_port ${VIZ_PORT}
