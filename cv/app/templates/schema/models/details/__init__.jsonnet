// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 18:49

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.string('network.checkpoints_root', 'Checkpoint Root', def='/cache', ddd=true, readonly=true),
                _Utils.string('network.checkpoints_dir', 'Checkpoint Path', def='ckpts', ddd=true, readonly=true),
                _Utils.string('_k12.network.pretrained_path', 'Pretrained Root', def='/pretrained', readonly=true),
            ],
        },
    ],
}
