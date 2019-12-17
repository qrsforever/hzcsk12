// @file random_brightness.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:02

local lib = import 'common.libsonnet';

{
    // string enum
    get(prefix): {
        name: { en: 'Random Brightness', cn: self.en },
        value: 'random_brightness',
        trigger: {
            type: 'object',
            objs: [
                lib.radio(prefix + '.aug_trans.random_brightness.ratio'),
                {
                    name: { en: 'shift_value', cn: self.en },
                    _id_: prefix + '.aug_trans.random_brightness.shift_value',
                    type: 'int',
                    default: 32,
                },
            ],
        },
    },
}
