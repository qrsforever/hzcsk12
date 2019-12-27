// @file bucket.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 00:09

{
    type: '_ignore_',
    objs: [
        {
            _id_: 'iterator.sorting_keys',
            name: { en: 'sorting keys', cn: self.en },
            type: 'string-array',
            default: [],
        },
        {
            _id_: 'iterator.padding_noise',
            name: { en: 'padding noise', cn: self.en },
            min: 0.0001,
            max: 0.9999,
            type: 'float',
            default: 0.1,
        },
        {
            _id_: 'iterator.biggest_batch_first',
            name: { en: 'biggest batch first', cn: self.en },
            type: 'bool',
            default: false,
        },
        {
            _id_: 'iterator.skip_smaller_batches',
            name: { en: 'skip smaller batches', cn: self.en },
            type: 'bool',
            default: false,
        },
    ] + import 'common.libsonnet',
}
