// @file step.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 18:00

local lib = import 'common.libsonnet';

{
    local this = self,
    _id_:: 'solver.lr.step',
    name: { en: 'Learning Rate Policy: Step', cn: self.en },
    type: 'object',
    objs: [
        lib.gamma(this._id_ + '.gamma'),
        {
            _id_: this._id_ + '.step_size',
            name: { en: 'step size', cn: self.en },
            type: 'int',
            default: 50,
        },
    ],
}
