// @file predict.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-26 20:19

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: if _Utils.task == 'cls' then [
            _Utils.bool('metrics.predict_images', 'images result', def=false),
            _Utils.bool('metrics.predict_probs', 'target probs', def=false),
        ] else [],
    },
]
