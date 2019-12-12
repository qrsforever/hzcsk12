// @file multistep.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 18:00

local lib = import 'common.libsonnet';

{
    local this = self,
    _id_:: 'solver.lr.multistep',
    name: { en: 'Learning Rate Policy: MultiStep', cn: self.en },
    type: 'object',
    objs: ['gamma', 'step_size'],
    gamma: lib.gamma(this._id_ + '.gamma'),
    stepvalue: {
        _id_: this._id_ + '.stepvalue',
        name: { en: 'multistep size', cn: self.en },
        type: 'int-array',
        minnum: 2,
        default: [90, 120],
    },
}
