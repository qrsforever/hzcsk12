// @file adam.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 22:27

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        name: { en: 'Adam Parameters', cn: self.en },
        type: 'H',
        objs: [
            _Utils.float(jid + '.weight_decay', 'Decay', def=0.001),
            _Utils.floatarray(jid + '.betas', 'Betas', def=[0.5, 0.999]),
            _Utils.float(jid + '.eps', 'EPS', def=1e-8),
        ],
    },
}
