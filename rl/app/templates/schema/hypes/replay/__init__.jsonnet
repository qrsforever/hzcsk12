// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 09:32

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.int('algo.n_step_return', 'Return Step', def=1),
                _Utils.float('algo.replay_size', 'Replay Size', def=1e5, tips='todo, eat mem'),
                _Utils.float('algo.replay_ratio', 'Replay Ratio', def=8),
            ],
        },
    ] + if _Utils.network == 'dqn'
    then
        [import 'dqn.libsonnet']
    else [],
}
