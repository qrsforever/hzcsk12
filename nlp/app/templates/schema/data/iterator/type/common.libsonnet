// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 23:51

[
    {
        // The size of each batch of instances yielded when calling the iterator.
        _id_: 'iterator.batch_size',
        name: { en: 'batch size', cn: self.en },
        type: 'int',
        min: 8,
        max: 2048,
        default: 32,
    },
    {
        _id_: 'iterator.instances_per_epoch',
        name: { en: 'instances/epoch', cn: self.en },
        type: 'int',
        min: 8,
        max: 2048,
        default: 32,
    },
    {
        _id_: 'iterator.max_instances_in_memory',
        name: { en: 'max instances', cn: self.en },
        type: 'int',
        min: 8,
        max: 2048,
        default: 32,
    },
    {
        _id_: 'iterator.cache_instances',
        name: { en: 'cache instances', cn: self.en },
        type: 'bool',
        default: false,
    },
    {
        _id_: 'iterator.cache_instances',
        name: { en: 'cache instances', cn: self.en },
        type: 'bool',
        default: false,
    },
]
