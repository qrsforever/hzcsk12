// @file dqn.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 09:43

local _Utils = import '../../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.int('algo.n_step_return', 'Return Step', def=1),
            _Utils.float('algo.replay_size', 'Replay Size', def=1e5, tips='todo, eat mem'),
            _Utils.float('algo.replay_ratio', 'Replay Ratio', def=8),
            {
                _id_: 'algo.prioritized_replay',
                name: { en: 'Replay Prior', cn: self.en },
                type: 'bool-trigger',
                objs: [
                    {
                        value: true,
                        trigger: {
                            type: 'H',
                            objs: [
                                _Utils.float('algo.pri_alpha', 'Pri Alpha', def=0.6),
                                _Utils.float('algo.pri_beta_init', 'Pri Beta Init', def=0.6),
                                _Utils.float('algo.pri_beta_final', 'Pri Beta Final', def=1.0),
                                _Utils.float('algo.pri_beta_steps', 'Pri Beta Steps', def=50e6),
                                _Utils.float('algo.default_priority', 'Pri Default', def=1.0),
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

]
