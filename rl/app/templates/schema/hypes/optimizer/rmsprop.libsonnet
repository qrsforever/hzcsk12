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
                    _Utils.float('optim.alpha',
                                 'Alpha',
                                 min=0.0,
                                 max=1.0,
                                 def=0.99,
                                 tips='smoothing constant'),
                    _Utils.float('optim.eps',
                                 'Eps',
                                 def=1e-8,
                                 tips='term added to the denominator to improve numerical stability'),
                ],
            },
            {
                type: 'H',
                objs: [
                    _Utils.float('optim.weight_decay',
                                 'Weight Decay',
                                 min=0.0,
                                 max=1.0,
                                 def=0.0,
                                 tips='weight decay (L2 penalty)'),
                    _Utils.float('optim.momentum',
                                 'Momentum',
                                 min=0.0,
                                 max=1.0,
                                 def=0.0,
                                 tips='momentum factor'),
                ],
            },
        ],
    },
}
