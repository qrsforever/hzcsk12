// @file dpn.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 00:39

local _Utils = import '../../utils/helper.libsonnet';

{
    type: 'V',
    objs: [
        {
            _id_: '_k12.model.name',
            name: { en: 'Model', cn: self.en },
            type: 'string-enum-trigger',
            objs: [
                {
                    name: { en: 'DQN', cn: self.en },
                    value: 'dqn',
                    trigger: {},
                },
                {
                    name: { en: 'CatDQN', cn: self.en },
                    value: 'catdqn',
                    trigger: {},
                },
                {
                    name: { en: 'R2D1', cn: self.en },
                    value: 'r2d1',
                    trigger: {},
                },
            ],
            default: self.objs[0].value,
        },
        {
            type: 'H',
            objs: [
                _Utils.int('model.fc_sizes', 'FC Size', def=512, readonly=true),
                _Utils.intarray('model.channels', 'Channels', def=[32, 64, 64]),
                _Utils.intarray('model.kernel_sizes', 'Kernels', def=[8, 4, 3]),
                _Utils.intarray('model.strides', 'Strides', def=[4, 2, 1]),
                _Utils.intarray('model.paddings', 'Paddings', def=[0, 1, 1]),
                _Utils.bool('model.use_maxpool', 'Use MaxPool', def=false),
            ],
        },
    ],
}
