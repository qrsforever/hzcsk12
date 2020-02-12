// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 15:36

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.string('_k12.data.dataset_name', 'Dataset Name', def=_Utils.dataset_name, readonly=true),
                _Utils.string('data.data_dir', 'Dataset Path', def='unkown', ddd=true, readonly=true),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('data.num_instances', 'Num Instances', def=0, ddd=true, readonly=true),
                _Utils.int('data.num_features', 'Num Features', def=0, ddd=true, readonly=true),
                _Utils.int('data.num_classes', 'Num Classes', def=0, ddd=true, readonly=true),
            ],
        },
    ],
}
