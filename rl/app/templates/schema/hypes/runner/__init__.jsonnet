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
            _Utils.int('affinity.cuda_idx', 'CUDA Device', def=0),
            _Utils.int('runner.n_steps', 'Iters Count', def=20e6),
            _Utils.int('runner.log_interval_steps', 'Summary Interval', def=2e5),
        ],
    },
}
