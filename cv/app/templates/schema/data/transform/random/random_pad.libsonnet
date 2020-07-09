// @file random_pad.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 19:58

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.ratio', 'ratio', def=0.5),
        _Utils.floatarray(jid + '.up_scale_range', 'scale range', def=[0, 1]),
        _Utils.intarray(jid + '.mean', 'mean', def=[104, 117, 124]),
    ],
}
