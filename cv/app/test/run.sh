#!/bin/bash

cur_dir=`pwd`

# 环境测试脚本

python -m torch.distributed.launch --nproc_per_node=1 \
    /hzcsk12/torchcv/main.py --dist y --phase train \
    --config_file $cur_dir/config.json \
    --checkpoints_root /tmp/
