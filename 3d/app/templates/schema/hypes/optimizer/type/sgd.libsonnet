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
            _Utils.float(jid + '.weight_decay',
                         'Decay',
                         min=0.0,
                         max=1.0,
                         def=0.001,
                         tips='weight decay (L2 penalty)'),
            _Utils.float(jid + '.momentum',
                         'Momentum',
                         min=0.0,
                         max=1.0,
                         def=0.9,
                         tips='momentum factor'),
            _Utils.bool(jid + '.nesterov',
                        'Nesterov',
                        def=false,
                        tips='whether using Nesterov momentum'),
        ],
    },
}
