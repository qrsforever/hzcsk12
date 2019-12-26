// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 15:23

local task = std.extVar('task');

[
    {
        _id_: '_k12.dataset.select',
        name: { en: 'Dataset', cn: self.en },
        type: 'string-enum-trigger',
        objs: if task == 'sa' then
            import 'task/sa.libsonnet',
        //     {
        //         name: { en: '', cn: self.en },
        //         value: '',
        //         trigger: {
        //             objs: [
        //                 {
        //                     _id_: 'xx',
        //                     name: { en: '', cn: self.en },
        //                     type: 'int',
        //                     default: 100,
        //                 },
        //             ],
        //         },
        //     },
        // ],
    },
    {
        name: { en: 'Reader', cn: self.en },
        type: 'navigation',
        objs: [
            {
                name: { en: 'Train', cn: self.en },
                objs: [
                ],
            },
            {
                name: { en: 'Validation', cn: self.en },
                objs: [
                ],
            },
        ],
    },
]
