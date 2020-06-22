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
            _Utils.int('solver.max_epoch', 'Max Epoch', def=100, ddd=true),
        ],
    },
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'LR', cn: self.en },
                type: '_ignore_',
                objs: [(import 'lr/__init__.jsonnet').get('solver.lr')],
            },
            {
                name: { en: 'Loss', cn: self.en },
                type: '_ignore_',
                objs: [(import 'loss/__init__.jsonnet').get('loss')],
            },
            {
                name: { en: 'Optimizer', cn: self.en },
                type: '_ignore_',
                objs: [(import 'optimizer/__init__.jsonnet').get('solver.optim')],
            },
        ],
    },

]
