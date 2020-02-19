// @file dqn.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 08:33

local _Utils = import '../../utils/helper.libsonnet';

{
    type: '_ignore_',
    objs: [
        {
            type: 'H',
            objs: [
                _Utils.int('algo.batch_size', 'Batch Size', def=32),
                _Utils.int('algo.min_steps_learn', 'Min Steps', def=5e4, tips='min steps learn'),
                _Utils.int('algo.delta_clip', 'Delta Clip', def=1.0),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('algo.target_update_tau', 'Update Tau', def=1),
                _Utils.int('algo.target_update_interval', 'Update Interval', def=312),
                _Utils.int('algo.eps_steps', 'Eps Steps', def=1e6),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('agent.eps_init', 'Eps Init', def=1.0),
                _Utils.float('agent.eps_final', 'Eps Final', def=0.01),
                _Utils.float('agent.eps_eval', 'Eps Eval', def=0.001),
                {
                    _id_: '_k12.agent.eps_final_min',
                    name: { en: 'Enable Eps Final Min', cn: self.en },
                    type: 'bool-trigger',
                    objs: [
                        {
                            value: true,
                            trigger: {
                                type: '_ignore_',
                                objs: [
                                    _Utils.float('agent.eps_final_min', 'Min', def=0.0005),
                                ],
                            },
                        },
                        {
                            value: false,
                            trigger: {},
                        },
                    ],
                    default: false,
                },
            ],
        },
    ],
}
