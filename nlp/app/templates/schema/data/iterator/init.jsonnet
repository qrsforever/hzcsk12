// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 00:35

{
    get(dataset): {
        _id_: 'iterator.type',
        name: { en: 'iterator type', cn: self.en },
        type: 'string-enum-trigger',
        objs: [
            {
                name: { en: 'bucket', cn: self.en },
                value: 'bucket',
                trigger: (import 'type/bucket.libsonnet').get(dataset),
            },
            {
                name: { en: 'basic', cn: self.en },
                value: 'basic',
                trigger: (import 'type/basic.libsonnet').get(dataset),
            },
        ],
        default: self.objs[0].value,
    },
}
