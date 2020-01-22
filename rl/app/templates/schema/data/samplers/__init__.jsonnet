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
                _Utils.int('sampler.batch_T', 'Batch T', def=32, tips='Number of time steps'),
                _Utils.int('sampler.batch_B', 'Batch B', def=16, tips='Number of separate trajectory segments'),
                _Utils.int('sampler.max_decorrelation_steps', 'Decorr Steps', def=100),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('sampler.eval_n_envs', 'Num Envs', def=2, min=1),
                _Utils.int('sampler.eval_max_steps', 'Max Steps', def=125e3),
                _Utils.int('sampler.eval_max_trajectories', 'Max Trajectories', def=100),
            ],
        },
        _Utils.bool('_k12.sampler.mid_batch_reset', 'Mid Batch Reset', def=false, tips='whether  environment resets during itr'),
    ],
}
