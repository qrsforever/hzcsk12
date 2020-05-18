// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 08:13

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: {
        type: 'H',
        objs: [
            _Utils.int('runner.n_steps', 'Num Steps', def=20e4, min=10e4),
            _Utils.int('runner.log_interval_steps', 'Log Interval Steps', def=2e3, min=1e3),
            // _Utils.bool('_k12.runner.eval', 'Evaluate', def=false),
        ],
    },
}
