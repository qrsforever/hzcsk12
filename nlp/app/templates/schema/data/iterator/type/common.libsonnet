// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 23:51

local _BASIC = import '../../../utils/basic_type.libsonnet';

[
    _BASIC.int('iterator.batch_size', 'batch size', min=8, def=32),
    _BASIC.int('iterator.instances_per_epoch', 'instance per epoch', min=8, def=32),
    _BASIC.int('iterator.max_instances_in_memory', 'max instance', min=8, def=32),
    _BASIC.bool('iterator.cache_instances', 'cache instances', def=false),
    _BASIC.bool('iterator.track_epoch', 'track epoch', def=false),
]
