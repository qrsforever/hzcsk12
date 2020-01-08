// @file random_focus_crop.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 19:52

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.radio', 'radio', def=0.5),
        _Utils.floatarray(jid + '.crop_size', 'crop size', def=[32, 32]),
        _Utils.int(jid + '.center_jitter', 'center jitter', def=0),
        _Utils.intarray(jid + '.mean', 'mean', def=[104, 117, 124]),
        _Utils.bool(jid + '.allow_outsize_center', 'outsize center', def=true),
    ],
}
