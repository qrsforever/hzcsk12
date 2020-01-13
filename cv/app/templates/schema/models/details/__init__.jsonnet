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
                _Utils.string('network.checkpoints_root', 'Checkpoint Root', def='/cache', ddd=true, readonly=true, width=400),
                _Utils.string('network.checkpoints_dir', 'Checkpoint Path', def='ckpts', ddd=true, readonly=true, width=400),
                _Utils.string('_k12.network.pretrained_path', 'Pretrained Root', def='/pretrained', readonly=true, width=400),
                _Utils.intarray('anchor.num_anchor_list', 'Anchor List', def=[], ddd=true, readonly=true, width=400),
                _Utils.intarray('anchor.cur_anchor_sizes', 'Anchor Sizes', def=[], ddd=true, readonly=true, width=400),
                _Utils.intarray('anchor.feature_maps_wh', 'Feature Maps', def=[], ddd=true, readonly=true, width=400),
                _Utils.intarray('anchor.aspect_ratio_list', 'Ratio List', def=[], ddd=true, readonly=true, width=400),
                _Utils.intarray('network.num_feature_list', 'Feature List', def=[], ddd=true, readonly=true, width=400),
                _Utils.intarray('network.stride_list', 'Stride List', def=[], ddd=true, readonly=true, width=400),
                _Utils.intarray('network.head_index_list', 'Head Index List', def=[], ddd=true, readonly=true, width=400),
            ],
        },
    ],
}
