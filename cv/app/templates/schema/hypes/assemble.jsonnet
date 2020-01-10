// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 23:06

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            {
                _id_: 'solver.lr.metric',
                name: { en: 'Metric', cn: self.en },
                type: 'string-enum-trigger',
                objs: [
                    {
                        name: { en: 'epoch', cn: self.en },
                        value: 'epoch',
                        trigger: {
                            type: '_ignore_',
                            objs: [
                                _Utils.int('solver.max_epoch', 'Max Epoch', def=100, ddd=true),
                            ],
                        },
                    },
                    {
                        name: { en: 'iters', cn: self.en },
                        value: 'iters',
                        trigger: {
                            type: '_ignore_',
                            objs: [
                                _Utils.int('solver.max_iters', 'Max Iters', def=10000, ddd=true),
                            ],
                        },
                    },
                ],
                default: 'epoch',
            },
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
