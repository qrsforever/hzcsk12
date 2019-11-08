#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Generate train & val data.


ORI_ROOT_DIR='/home/dc2-user/work/qrs/datasets/ade20k/ADEChallengeData2016'
SAVE_DIR='/home/dc2-user/work/qrs/datasets/ade20k'


python ade20k_seg_generator.py --ori_root_dir $ORI_ROOT_DIR \
                               --save_dir $SAVE_DIR
