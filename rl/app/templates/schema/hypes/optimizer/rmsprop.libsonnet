// @file rmsprop.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-20 18:19

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: {
        name: { en: 'Rmsprop Parameters', cn: self.en },
        type: '_ignore_',
        objs: [
            {
                type: 'H',
                objs: [
                    _Utils.float('optim.alpha', 'Alpha', def=0.99),
                    _Utils.float('optim.eps', 'Eps', def=1e-8),
                ],
            },
            {
                type: 'H',
                objs: [
                    _Utils.float('optim.weight_decay', 'Weight Decay', def=0.0),
                    _Utils.float('optim.momentum', 'Momentum', def=0.0),
                ],
            },
        ],
    },
}
