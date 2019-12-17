// @file random_saturation.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:52

local lib = import 'common.libsonnet';

{
    random(prefix): {
        local this = self,
        _id_:: prefix + '.aug_trans.random_saturation',
        name: { en: 'Random Saturation Parameters', cn: self.en },
        type: 'object',
        objs: ['ratio', 'lower', 'upper'],
        ratio: lib.radio(this._id_ + '.ratio'),
        lower: lib.lower(this._id_ + '.lower'),
        upper: lib.upper(this._id_ + '.upper'),
    },
}
