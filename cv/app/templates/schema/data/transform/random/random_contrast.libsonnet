// @file random_contrast.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 18:54

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.ratio', 'ratio', def=0.5),
        _Utils.float(jid + '.lower', 'lower', def=0.5),
        _Utils.float(jid + '.upper', 'upper', def=0.5),
    ],
}
