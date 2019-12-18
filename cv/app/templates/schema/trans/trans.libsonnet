// @file trans.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-16 14:09

{
    get(prefix): {
        name: { en: '%s' % prefix, cn: self.en },
        type: 'HV',
        objs: [
            {
                _id_: prefix + '.batch_size',
                name: { en: 'batch size', cn: self.en },
                type: 'int',
                default: 128,
            },
            {
                name: { en: 'Random Transform Method', cn: self.en },
                type: 'string-enum-group-trigger',
                objs: [
                    (import 'random/random_contrast.libsonnet').get(prefix),
                    (import 'random/random_brightness.libsonnet').get(prefix),
                    (import 'random/random_hue.libsonnet').get(prefix),
                    (import 'random/random_perm.libsonnet').get(prefix),
                    (import 'random/random_saturation.libsonnet').get(prefix),
                ],
                groups: [
                    {
                        name: { en: 'None', cn: self.en },
                        value: 'none',
                    },
                    {
                        name: { en: 'Order', cn: self.en },
                        value: prefix + '.aug_trans.trans_seq',
                    },
                    {
                        name: { en: 'Shuffle', cn: self.en },
                        value: prefix + '.aug_trans.shuffle_trans_seq',
                    },
                ],
                default: [],
            },
        ],
    },
}
