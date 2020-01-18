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
                _Utils.int('algo.min_steps_learn', 'Min Steps', def=5e4),
                _Utils.int('algo.eps_steps', 'Eps Steps', def=1e6),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('algo.delta_clip', 'Delta Clip', def=1.0),
                _Utils.int('algo.target_update_tau', 'Update Tau', def=1),
                _Utils.int('algo.target_update_interval', 'Update Interval', def=312),
            ],
        },
    ],
}
