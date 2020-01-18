// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 08:20

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.float('algo.discount', 'Discount', def=0.99),
                _Utils.float('algo.learning_rate', 'LR', def=1e-4),
                _Utils.float('algo.clip_grad_norm', 'Clip Grad Norm', def=10.0),
            ],
        },
    ] + if _Utils.network == 'dqn'
    then
        [import 'dqn.libsonnet']
    else [],
}
