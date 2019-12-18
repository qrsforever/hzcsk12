// @file random_brightness.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:02

local lib = import 'common.libsonnet';

{
    get(prefix): {
        local this = self,
        _id_:: prefix + '.aug_trans.random_brightness',
        name: { en: 'Brightness', cn: self.en },
        value: 'random_brightness',
        trigger: {
            name: this.name,
            type: 'object',
            objs: [
                lib.radio(this._id_ + '.ratio'),
                {
                    name: { en: 'shift_value', cn: self.en },
                    _id_: this._id_ + '.shift_value',
                    type: 'int',
                    default: 32,
                },
            ],
        },
    },
}
