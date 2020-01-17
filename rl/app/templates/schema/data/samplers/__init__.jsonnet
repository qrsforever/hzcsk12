// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-17 15:49

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.int('sampler.batch_T', 'Batch T', def=32),
                _Utils.int('sampler.batch_B', 'Batch B', def=16),
                _Utils.int('sampler.max_decorrelation_steps', 'Decorr Steps', def=100),
            ],
        },
        {
            _id_: '_k12.sampler.eval',
            name: { en: 'Eval Enable', cn: self.en },
            type: 'bool-trigger',
            objs: [
                {
                    value: true,
                    trigger: {
                        type: '_ignore_',
                        objs: [
                            {
                                type: 'H',
                                objs: [
                                    _Utils.int('sampler.eval_n_envs', 'Num Envs', def=4, min=1),
                                    _Utils.int('sampler.eval_max_steps', 'Max Steps', def=125e3),
                                    _Utils.int('sampler.eval_max_trajectories', 'Max Trajectories', def=100),
                                ],
                            },
                        ],
                    },
                },
                {
                    value: true,
                    trigger: {},
                },
            ],
            default: false,
        },
    ],
}
