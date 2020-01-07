// @file random_det_crop.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 19:55

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.radio', 'radio', def=0.5),
    ],
}
