// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-27 15:04

local _Utils = import '../../utils/helper.libsonnet';

local _READERS = {
    sst: {
        get(jid, navi): [
            {
                name: { en: 'SST Tokens', cn: self.en },
                value: 'sst_tokens',
                trigger: {
                    objs: (import 'sst.libsonnet').get(jid, navi),
                },
            },
        ],
    },  // dataset_name: sst
};

{
    get(dataset_name): {
        name: { en: 'Phase', cn: self.en },
        type: 'navigation',
        objs: [
            {
                local jid = 'dataset_name_reader',
                name: { en: 'Train', cn: self.en },
                objs: [
                    {
                        _id_: jid + '.type',
                        name: { en: 'Type', cn: self.en },
                        type: 'string-enum-trigger',
                        objs: _READERS[dataset_name].get(jid, 'train'),
                        default: self.objs[0].value,
                    },
                ],
            },
            {
                local jid = 'validation_dataset_name_reader',
                name: { en: 'Validation', cn: self.en },
                objs: [
                    {
                        _id_: '_k12.validation_dataset_name_reader.bool',
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
                                            objs: _READERS[dataset_name].get(jid, 'validation'),
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
