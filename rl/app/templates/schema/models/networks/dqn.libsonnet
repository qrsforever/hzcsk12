// @file dpn.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 00:39

local _Utils = import '../../utils/helper.libsonnet';

{
    type: '_ignore_',
    objs: [
        {
            _id_: '_k12.models.variant',
            name: { en: 'Variant', cn: self.en },
            type: 'string-enum-trigger',
            objs: [
                {
                    name: { en: 'None', cn: self.en },
                    value: 'none',
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
                _Utils.int('models.fc_sizes', 'FC Size', def=512, readonly=true),
                _Utils.intarray('models.channels', 'Channels', def=[32, 64, 64]),
                _Utils.intarray('models.kernel_sizes', 'Kernels', def=[8, 4, 3]),
                _Utils.intarray('models.strides', 'Strides', def=[4, 2, 1]),
                _Utils.intarray('models.paddings', 'Paddings', def=[0, 1, 1]),
                _Utils.bool('models.use_maxpool', 'Use MaxPool', def=false),
            ],
        },
    ],
}
