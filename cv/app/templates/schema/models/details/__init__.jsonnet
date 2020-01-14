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
            ] + if _Utils.network == 'vgg16_ssd300' then
                [
                    _Utils.intarray('anchor.num_anchor_list', 'Anchor List', def=[4, 6, 6, 6, 4, 4], readonly=true),
                    _Utils.intarray('anchor.cur_anchor_sizes', 'Anchor Sizes', def=[30, 60, 111, 162, 213, 264, 315], readonly=true),
                    _Utils.intarray('anchor.feature_maps_wh', 'Feature Maps', def=[[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]], readonly=true),
                    _Utils.intarray('anchor.aspect_ratio_list', 'Ratio List', def=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], readonly=true),
                    _Utils.intarray('network.num_feature_list', 'Feature List', def=[512, 1024, 512, 256, 256, 256], readonly=true),
                    _Utils.intarray('network.stride_list', 'Stride List', def=[8, 16, 30, 60, 100, 300], readonly=true),
                    _Utils.intarray('network.head_index_list', 'Head Index List', def=[0, 1, 2, 3, 4, 5], readonly=true),
                ] else if _Utils.network == 'vgg16_ssd512' then
                [
                    _Utils.intarray('anchor.num_anchor_list', 'Anchor List', def=[4, 6, 6, 6, 6, 4, 4], readonly=true),
                    _Utils.intarray('anchor.cur_anchor_sizes', 'Anchor Sizes', def=[35, 76, 153, 230, 307, 384, 460, 537], readonly=true),
                    _Utils.intarray('anchor.feature_maps_wh', 'Feature Maps', def=[[64, 64], [32, 32], [16, 16], [8, 8], [4, 4], [2, 2], [1, 1]], readonly=true),
                    _Utils.intarray('anchor.aspect_ratio_list', 'Ratio List', def=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]], readonly=true),
                    _Utils.intarray('network.num_feature_list', 'Feature List', def=[512, 1024, 512, 256, 256, 256, 256], readonly=true),
                    _Utils.intarray('network.stride_list', 'Stride List', def=[8, 16, 32, 64, 128, 256, 512], readonly=true),
                    _Utils.intarray('network.head_index_list', 'Head Index List', def=[0, 1, 2, 3, 4, 5, 6], readonly=true),
                ] else [],
        },
    ],
}
