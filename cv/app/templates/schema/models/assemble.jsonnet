// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 23:05

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.bool('network.distributed', 'Distributed', def=true),
            _Utils.bool('_k12.network.pretrained', 'Pretrained', def=true),
        ],
    },
    (import 'network/__init__.jsonnet').get(),
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Custom', cn: self.en },
                type: '_ignore_',
                objs: [
                    _Utils.text('_k12.notimpl.custom', 'Not Impl Yet', def='todo'),
                ],
            },
            {
                name: { en: 'Details', cn: self.en },
                type: '_ignore_',
                objs: [
                    (import 'details/__init__.jsonnet').get(),
                ],
            },
        ],
    },
]
