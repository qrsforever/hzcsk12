// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-27 15:04

local _BASIC = import '../../utils/basic_type.libsonnet';

local _READERS = {
    sst: {
        get(jid): [
            {
                name: { en: 'SST Tokens', cn: self.en },
                value: 'sst_tokens',
                trigger: {
                    objs: [
                        {
                            type: 'H',
                            objs: [
                                _BASIC.bool(jid + '.layz', 'lazy', def=true),
                                _BASIC.bool(jid + '.use_subtrees', 'use subtrees', def=false),
                                _BASIC.string(jid + '.granularity', 'granularity', def='5-class'),
                            ],
                        },
                    ],
                },
            },
        ],
    },  // dataset: sst
};

{
    get(dataset): {
        assert std.length(dataset) != 0,
        name: { en: 'Phase', cn: self.en },
        type: 'navigation',
        objs: [
            {
                local jid = 'dataset_reader',
                name: { en: 'Train', cn: self.en },
                objs: [
                    {
                        _id_: jid + '.type',
                        name: { en: 'Type', cn: self.en },
                        type: 'string-enum-trigger',
                        objs: _READERS[dataset].get(jid),
                        default: self.objs[0].value,
                    },
                ],
            },
            {
                local jid = 'validation_dataset_reader',
                name: { en: 'Validation', cn: self.en },
                objs: [
                    {
                        _id_: '_k12.validation.bool',
                        name: { en: 'Enable', cn: self.en },
                        type: 'bool-trigger',
                        objs: [
                            {
                                value: true,
                                trigger: {
                                    objs: [
                                        {
                                            _id_: jid + '.type',
                                            name: { en: 'Type', cn: self.en },
                                            type: 'string-enum-trigger',
                                            objs: _READERS[dataset].get(jid),
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
