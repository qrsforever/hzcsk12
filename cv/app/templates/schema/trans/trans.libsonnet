// @file trans.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-16 14:09

{
    get(prefix): {
        name: { en: '%s' % prefix, cn: self.en },
        type: 'object',
        objs: [
            {
                name: { en: 'random sequence', cn: self.en },
                type: 'string-enum-group-trigger',
                objs: [
                    (import 'random/random_contrast.libsonnet').get(prefix),
                    (import 'random/random_brightness.libsonnet').get(prefix),
                ],
                // button
                groups: [
                    {
                        name: { en: 'None', cn: self.en },
                        value: 'none',
                    },
                    {
                        name: { en: 'Normal', cn: self.en },
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
// train.aug_trans.trans_seq = [ ]

//            {
//                _id_: this._id_ + '.batch_size',
//                name: { en: 'batch size', cn: self.en },
//                type: 'int',
//                default: '128',
//            },



