// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 15:23

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.string('_k12.data.dataset_name',
                          'Dataset Name',
                          def=_Utils.dataset_name,
                          readonly=true,
                          tips='dataset name'),
            _Utils.bool('trainer.shuffle', 'Shuffle', def=true, ddd=true),
        ],
    },
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Reader', cn: self.en },
                type: '_ignore_',
                objs: [
                    (import 'readers/__init__.jsonnet').get(),
                ],
            },
            {
                name: { en: 'Iterator', cn: self.en },
                type: '_ignore_',
                objs: [
                    (import 'iterator/__init__.jsonnet').get(),
                ],
            },
        ],
        // + if _Utils.debug then [
        //     {
        //         name: { en: 'Debug', cn: self.en },
        //         type: '_ignore_',
        //         objs: [
        //             _Utils.text('_k12.dev',
        //                         'NB',
        //                         def=_Utils.notebook_url,
        //                         width=800,
        //                         readonly=true),
        //         ],
        //     },
        // ] else [],
    },
    // {
    //     type: 'H',
    //     objs: [
    //         _Utils.bool('_k12.dev_mode', 'Develop Mode', def=false),
    //     ],
    // },
]
