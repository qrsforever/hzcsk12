// @file random_hflip.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 20:04

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.ratio', 'ratio', def=0.5),
        _Utils.intarray(jid + '.swap_pair', 'swap pair', def=[[]]),
    ],
}
