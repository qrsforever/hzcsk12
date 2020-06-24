// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-22 18:02

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.int('hypes.epoch', 'Epoch', def=100, min=1),
        ],
    },
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Loss', cn: self.en },
                type: '_ignore_',
                objs: [(import 'loss/__init__.jsonnet').get('hypes.criterion')],
            },
            {
                name: { en: 'Optimizer', cn: self.en },
                type: '_ignore_',
                objs: [(import 'optimizer/__init__.jsonnet').get('hypes.optimizer')],
            },
            {
                name: { en: 'Scheduler', cn: self.en },
                type: '_ignore_',
                objs: [(import 'lr/__init__.jsonnet').get('hypes.scheduler')],
            },
        ],
    },

]
