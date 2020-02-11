// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 23:06

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.int('model.args.max_iter', 'degree', def=1000),
            _Utils.float('model.args.cache_size', 'cache_size', def=200.0),
        ],
    },
]
