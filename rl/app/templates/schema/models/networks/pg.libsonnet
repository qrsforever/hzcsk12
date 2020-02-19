// @file pg.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 00:38

local _Utils = import '../../utils/helper.libsonnet';

{
    type: 'V',
    objs: [
        _Utils.string('_k12.model.network', 'Network', def=_Utils.network, readonly=true),
        {
            type: 'H',
            objs: [
                {
                    _id_: '_k12.model.name',
                    name: { en: 'Model', cn: self.en },
                    type: 'string-enum-trigger',
                    objs: [
                        {
                            name: { en: 'A2C', cn: self.en },
                            value: 'a2c',
                            trigger: {},
                        },
                        {
                            name: { en: 'PPO', cn: self.en },
                            value: 'ppo',
                            trigger: {
                                type: 'V',
                                objs: [
                                    {
                                        type: 'H',
                                        objs: [
                                            _Utils.int('algo.epochs', 'Epochs', def=4, tips='PPO Used'),
                                            _Utils.int('algo.minibatches', 'Mini Batches', def=4, tips='PPO Used'),
                                        ],
                                    },
                                    {
                                        type: 'H',
                                        objs: [
                                            _Utils.float('algo.ratio_clip', 'Ratio Clip', def=0.1, tips='PPO Used'),
                                            _Utils.bool('alog.linear_lr_schedule', 'Linear LR Schedule', def=true, tips='PPO Used'),
                                            _Utils.bool('alog.normalize_advantage', 'Normal Advantage', def=false, tips='PPO Used'),
                                        ],
                                    },
                                ],
                            },
                        },
                    ],
                    default: self.objs[0].value,
                },
            ],
        },
    ],
}
