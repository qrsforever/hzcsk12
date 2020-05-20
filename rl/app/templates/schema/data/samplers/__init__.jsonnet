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
                _Utils.int('sampler.batch_T',
                           'Batch T',
                           min=1,
                           def=32,
                           tips='number of time-steps per sample batch'),
                _Utils.int('sampler.batch_B',
                           'Batch B',
                           min=1,
                           def=16,
                           tips='number of environment instances to run in parallel, becomes second batch dimension'),
                _Utils.int('sampler.max_decorrelation_steps',
                           'Decorr Steps',
                           def=100,
                           tips='number of steps before start of training, to decorrelate batch states'),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('sampler.eval_n_envs', 'Num Envs', def=1, min=1, readonly=true),
                _Utils.int('sampler.eval_max_steps',
                           'Max Steps',
                           def=125e3,
                           tips='max total number of steps per evaluation call'),
                _Utils.int('sampler.eval_max_trajectories',
                           'Max Trajectories',
                           def=100,
                           tips='earlier cutoff for evaluation phase'),
            ],
        },
        _Utils.bool('_k12.sampler.mid_batch_reset', 'Mid Batch Reset', def=true, readonly=true, tips='whether environment resets during itr'),
    ],
}
