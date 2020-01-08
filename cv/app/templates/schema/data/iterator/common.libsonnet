// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 15:04

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid)::
        [
            _Utils.int(jid + '.batch_size', 'Batch Size', min=8, def=32),
        ],
}
