// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-17 16:10

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'V',
        objs: (import 'affinity/__init__.jsonnet').get(),
    },
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Environment', cn: self.en },
                type: '_ignore_',
                objs: (import 'envs/__init__.jsonnet').get(),
            },
            {
                name: { en: 'Sampler', cn: self.en },
                type: '_ignore_',
                objs: (import 'samplers/__init__.jsonnet').get(),
            },
        ],
    },
]
