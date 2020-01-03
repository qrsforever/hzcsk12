// @file sgd.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 12:09

// lr (float): learning rate
// momentum (float, optional): momentum factor (default: 0)
// weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'H',
        objs: [
            _Utils.float(jid + '.momentum', 'Momentum', def=0),
        ] + (import 'common.libsonnet').get(jid),
    },
}
