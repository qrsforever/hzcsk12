// @file multistep.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 18:00

local lib = import 'common.libsonnet';

{
    local this = self,
    _id_:: 'solver.lr.multistep',
    type: 'object',
    objs: [
        lib.gamma(this._id_ + '.gamma'),
        {
            _id_: this._id_ + '.stepvalue',
            name: { en: 'Multi Size', cn: self.en },
            type: 'int-array',
            minnum: 2,
            default: [90, 120],
        },
    ],
}
