// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 12:07

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: [
        _Utils.float(jid + '.lr', 'lr', def=1e-3),
        _Utils.float(jid + '.weight_decay', 'weight decay', def=0),
    ],
}
