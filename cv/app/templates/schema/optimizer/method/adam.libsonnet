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
    objs: [
        lib.weight_decay(this._id_ + '.weight_decay'),
        {
            _id_: this._id_ + '.betas',
            name: { en: 'Betas', cn: self.en },
            type: 'float-array',
            minnum: 2,
            maxnum: 2,
            default: [0.5, 0.999],
        },
        {
            _id_: this._id_ + '.eps',
            name: { en: 'EPS', cn: self.en },
            type: 'float',
            default: 1e-08,
        },
    ],
}
