// @file random_rotate.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 19:42

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.ratio', 'ratio', def=0.5),
        _Utils.int(jid + '.max_degree', 'degree', def=0),
        _Utils.intarray(jid + '.mean', 'mean', def=[104, 117, 124]),
    ],
}
