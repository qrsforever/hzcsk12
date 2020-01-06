// @file random_perm.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 19:16

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.ratio', 'ratio', def=0.5),
    ],
}
