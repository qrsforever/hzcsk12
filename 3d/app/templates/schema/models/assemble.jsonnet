// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-22 17:59

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.string('model.network', 'Network', def=_Utils.network, readonly=true),
            _Utils.bool('model.distributed', 'Distributed', def=false, readonly=true),
            _Utils.bool('model.resume',
                        'Resume',
                        def=false,
                        tips='continue with the last training'),
        ],
    },
    {
        type: 'H',
        objs: [
            _Utils.string('model.backbone', 'Backbone', def=_Utils.backbone, readonly=true),
            _Utils.bool('model.pretrained',
                        'Pretrained',
                        def=false,
                        tips='if true using the pretrained models weights, not support custom model'),
        ],
    },
]
