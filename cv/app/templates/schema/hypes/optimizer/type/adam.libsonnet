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
            _Utils.float(jid + '.weight_decay',
                         'Decay',
                         min=0.0,
                         max=1.0,
                         def=0.001,
                         tips='weight decay (L2 penalty)'),
            _Utils.floatarray(jid + '.betas',
                              'Betas',
                              def=[0.5, 0.999],
                              tips='value like [float, float], every element range [0.0, 1)'),
            _Utils.float(jid + '.eps',
                         'EPS',
                         def=1e-8,
                         tips='term added to the denominator to improve numerical stability',
                         readonly=true),
        ],
    },
}
