// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 00:35

{
    _id_: 'iterator.type',
    name: { en: 'iterator type', cn: self.en },
    type: 'string-enum-trigger',
    objs: [
        {
            name: { en: 'basic', cn: self.en },
            value: 'basic',
            trigger: import 'type/basic.libsonnet',
        },
        {
            name: { en: 'bucket', cn: self.en },
            value: 'bucket',
            trigger: import 'type/bucket.libsonnet',
        },
    ],
    default: 'bucket',
}
