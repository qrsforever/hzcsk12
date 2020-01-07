// @file random_border.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 20:00

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.radio', 'radio', def=0.5),
        _Utils.intarray(jid + '.pad', 'pad', def=[0, 0, 0, 0]),
        _Utils.intarray(jid + '.mean', 'mean', def=[104, 117, 124]),
        _Utils.bool(jid + '.allow_outsize_center', 'outsize center', def=true),
    ],
}
