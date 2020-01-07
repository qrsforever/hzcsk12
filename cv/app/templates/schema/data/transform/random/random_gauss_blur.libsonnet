// @file random_gaussblur.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 19:15

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.ratio', 'ratio', def=0.5),
        _Utils.int(jid + '.max_blur', 'max blur', def=4),
    ],
}
