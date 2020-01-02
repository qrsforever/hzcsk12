// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 00:35

{
    get(dataset_name): {
        name: { en: 'Phase', cn: self.en },
        type: 'navigation',
        objs: [
            {
                local jid = 'iterator',
                name: { en: 'Train', cn: self.en },
                objs: [
                    {
                        _id_: jid + '.type',
                        name: { en: 'Iterator Type', cn: self.en },
                        type: 'string-enum-trigger',
                        objs: [
                            {
                                name: { en: 'bucket', cn: self.en },
                                value: 'bucket',
                                trigger: (import 'type/bucket.libsonnet').get(jid, dataset_name),
                            },
                            {
                                name: { en: 'basic', cn: self.en },
                                value: 'basic',
                                trigger: (import 'type/basic.libsonnet').get(jid, dataset_name),
                            },
                        ],
                        default: self.objs[0].value,
                    },
                ],
            },
            {
                local jid = 'validation_iterator',
                name: { en: 'Validation', cn: self.en },
                objs: [
                    {
                        _id_: '_k12.validation_iterator.bool',
                        name: { en: 'Enable', cn: self.en },
                        type: 'bool-trigger',
                        objs: [
                            {
                                value: true,
                                trigger: {
                                    objs: [
                                        {
                                            _id_: jid + '.type',
                                            name: { en: 'Iterator Type', cn: self.en },
                                            type: 'string-enum-trigger',
                                            objs: [
                                                {
                                                    name: { en: 'bucket', cn: self.en },
                                                    value: 'bucket',
                                                    trigger: (import 'type/bucket.libsonnet').get(jid, dataset_name),
                                                },
                                                {
                                                    name: { en: 'basic', cn: self.en },
                                                    value: 'basic',
                                                    trigger: (import 'type/basic.libsonnet').get(jid, dataset_name),
                                                },
                                            ],
                                            default: self.objs[0].value,
                                        },
                                    ],
                                },
                            },
                            {
                                value: false,
                                trigger: {},
                            },
                        ],
                        default: false,
                    },
                ],
            },
        ],
    },
}
