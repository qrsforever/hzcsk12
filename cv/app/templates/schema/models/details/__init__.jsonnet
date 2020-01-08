// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 18:49

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: {
        type: '_ignore_',
        objs: [
            {
                type: 'H',
                objs: [
                    _Utils.string('network.checkpoints_root', 'Checkpoint Root', def='/hzcsk12/cv/data/cache', ddd=true, readonly=true),
                    _Utils.string('network.checkpoints_dir', 'Checkpoint Dir', def='ckpts', ddd=true, readonly=true),
                ],
            },
        ],
    },
}
