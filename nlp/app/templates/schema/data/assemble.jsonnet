// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 15:23

local _Utils = import '../utils/helper.libsonnet';

[
    _Utils.string('_k12.dataset', 'Dataset Name', def=_Utils.dataset_name, readonly=true),
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
    },
]
