// @file adam.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 11:44

local lib = import 'common.libsonnet';

{
    local this = self,
    _id_:: 'solver.optim.adam',
    name: { en: 'Adam Parameters', cn: self.en },
    type: 'object',
    objs: ['weight_decay', 'betas', 'eps'],
    weight_decay: lib.weight_decay(this._id_ + '.weight_decay'),
    betas: {
        _id_: this._id_ + '.betas',
        type: 'float-array',
        name: 'Betas',
        minnum: 2,
        maxnum: 2,
        default: [0.5, 0.999],
    },
    eps: {
        _id_: this._id_ + '.eps',
        type: 'float',
        name: 'EPS',
        default: 1e-08,
    },
}
