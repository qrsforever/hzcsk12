// @file random_contrast.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:38

local lib = import 'common.libsonnet';

{
    // string enum
    get(prefix): {
        name: { en: 'Random Contrast', cn: self.en },
        value: 'random_contrast',
        trigger: {
            type: 'object',
            objs: [
                lib.radio(prefix + '.aug_trans.random_contrast.ratio'),
                lib.lower(prefix + '.aug_trans.random_contrast.lower'),
                lib.upper(prefix + '.aug_trans.random_contrast.upper'),
            ],
        },
    },
}
