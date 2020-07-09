// @file train.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-26 20:17

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: if _Utils.task == 'cls' then [
            _Utils.bool('metrics.raw_vs_aug', 'Raw & Aug', def=false),
            _Utils.bool('metrics.train_speed', 'Train Speed', def=false),
            _Utils.bool('metrics.train_lr', 'Learn Rate', def=false),
            _Utils.bool('metrics.val_speed', 'Valid Speed', def=false),
        ] else [],
    },
]
