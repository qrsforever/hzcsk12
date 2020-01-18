// @file adam.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 09:05

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: {
        name: { en: 'Adam Parameters', cn: self.en },
        type: 'H',
        objs: [
            _Utils.floatarray('algo.optim_kwargs.betas', 'Betas', def=[0.9, 0.999]),
            _Utils.float('algo.optim_kwargs.eps', 'Eps', def=1e-8),
            _Utils.float('algo.optim_kwargs.weight_decay', 'Weight Decay', def=0),
        ],
    },
}
