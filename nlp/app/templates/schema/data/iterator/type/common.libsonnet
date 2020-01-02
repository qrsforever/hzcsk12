// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 23:51

local _BASIC = import '../../../utils/basic_type.libsonnet';

{
    get(jid):
        [
            _BASIC.int(jid + '.batch_size', 'batch size', min=8, def=32),
            _BASIC.int(jid + '.instances_per_epoch', 'instance per epoch', min=8, def=32),
            _BASIC.int(jid + '.max_instances_in_memory', 'max instance', min=8, def=32),
            _BASIC.bool(jid + '.cache_instances', 'cache instances', def=false),
            _BASIC.bool(jid + '.track_epoch', 'track epoch', def=false),
        ],
}
