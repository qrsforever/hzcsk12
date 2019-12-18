// @file random_saturation.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:52

local lib = import 'common.libsonnet';

{
    get(prefix): {
        local this = self,
        _id_:: prefix + '.aug_trans.random_saturation',
        name: { en: 'Saturation', cn: self.en },
        value: 'random_saturation',
        trigger: {
            name: this.name,
            type: 'object',
            objs: [
                lib.radio(this._id_ + '.ratio'),
                lib.lower(this._id_ + '.lower'),
                lib.upper(this._id_ + '.upper'),
            ],
        },
    },
}
