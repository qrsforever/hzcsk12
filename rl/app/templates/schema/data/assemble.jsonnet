// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-17 16:10

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.bool('_k12.sampler.async', 'Sampler Async', def=false),
            {
                _id_: '_k12.sample.mode',
                name: { en: 'Sampler Mode', cn: self.en },
                type: 'string-enum',
                objs: [
                    {
                        name: { en: 'cpu', cn: self.en },
                        value: 'cpu',
                    },
                    {
                        name: { en: 'gpu', cn: self.en },
                        value: 'gpu',
                    },
                    {
                        name: { en: 'alternating', cn: self.en },
                        value: 'alternating',
                    },
                    {
                        name: { en: 'serial', cn: self.en },
                        value: 'serial',
                    },
                ],
                default: self.objs[0].value,
            },
        ],
    },
    // {
    //     type: 'accordion',
    //     objs: [
    //         {
    //             name: { en: 'Affinity', cn: self.en },
    //             type: '_ignore_',
    //             objs: (import 'affinity/__init__.jsonnet').get(),
    //         },
    //         {
    //             name: { en: 'Environment', cn: self.en },
    //             type: '_ignore_',
    //             objs: (import 'envs/__init__.jsonnet').get(),
    //         },
    //         {
    //             name: { en: 'Sampler', cn: self.en },
    //             type: '_ignore_',
    //             objs: (import 'samplers/__init__.jsonnet').get(),
    //         },
    //     ],
    // },
]
