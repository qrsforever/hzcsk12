// @file random_resized_crop.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 19:37

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.intarray(jid + '.crop_size', 'crop size', def=[32, 32]),
        _Utils.floatarray(jid + '.scale_range', 'scale range', def=[0.01, 0.8]),
        _Utils.floatarray(jid + '.aspect_range', 'aspect range', def=[0.75, 1.33]),
    ],
}
