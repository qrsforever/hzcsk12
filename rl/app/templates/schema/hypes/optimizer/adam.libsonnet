// @file adam.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 09:05

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: {
        name: { en: 'Adam Parameters', cn: self.en },
        type: '_ignore_',
        objs: [
            {
                type: 'H',
                objs: [
                    _Utils.floatarray('optim.betas', 'Betas', def=[0.9, 0.999]),
                    _Utils.float('optim.eps', 'Eps', def=1e-8),
                ],
            },
            {
                type: 'H',
                objs: [
                    _Utils.float('optim.weight_decay', 'Weight Decay', def=0),
                    _Utils.bool('optim.amsgrad', 'Amsgrad', def=false),
                ],
            },
        ],
    },
}
