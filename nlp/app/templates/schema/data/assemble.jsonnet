// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 15:23

local _ITEM_FOR_TEST = {
    name: { en: 'TEST', cn: self.en },
    value: 'TEST',
    trigger: {
        objs: [
            {
                name: { en: 'TEST', cn: self.en },
                type: 'text',
                width: 300,
                height: 120,
                default: 'Only for test',
            },
        ],
    },
};

local _DATASETS = {
    sentiment_analysis: {
        get(): [
            {
                name: { en: 'SST', cn: self.en },
                value: 'sst',
                trigger: {
                    objs: [
                        (import 'readers/init.jsonnet').get('sst'),
                        (import 'iterator/init.jsonnet').get('sst'),
                    ],
                },
            },
            _ITEM_FOR_TEST,
        ],
    },  // sentiment_analysis

    semantic_role_labeling: [
        _ITEM_FOR_TEST,
    ],  // semantic_role_labeling

    reading_comprehension: [
        _ITEM_FOR_TEST,
    ],  // reading_comprehension
};

[
    local task = std.extVar('task');
    {
        _id_: '_k12.dataset.select',
        name: { en: 'Dataset', cn: self.en },
        type: 'string-enum-trigger',
        objs: _DATASETS[task].get(),
        default: self.objs[0].value,
    },
]
