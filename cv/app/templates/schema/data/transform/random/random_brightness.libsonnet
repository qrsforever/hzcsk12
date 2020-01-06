// @file random_brightness.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 18:33

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.ratio', 'ratio', def=0.5),
        _Utils.int(jid + '.shift_value', 'shift value', def=32),
    ],
}
