// @file random_hue.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:21

local lib = import 'common.libsonnet';

{
    get(prefix): {
        local this = self,
        _id_:: prefix + '.aug_trans.random_hue',
        name: { en: 'Hue', cn: self.en },
        value: 'random_hue',
        trigger: {
            name: this.name,
            type: 'object',
            objs: [
                lib.radio(this._id_ + '.ratio'),
                {
                    _id_: this._id_ + '.delta',
                    name: { en: 'delta', cn: self.en },
                    type: 'int',
                    default: 18,
                },
            ],
        },
    },
}
