// @file sgd.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 22:37

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        name: { en: 'SGD Parameters', cn: self.en },
        type: 'H',
        objs: [
            _Utils.float(jid + '.weight_decay', 'Decay', def=0.001),
            _Utils.float(jid + '.momentum', 'Momentum', def=0.9),
            _Utils.bool(jid + '.nesterov', 'Nesterov', def=false),
        ],
    },
}
