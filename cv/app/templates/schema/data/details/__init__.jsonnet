// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 15:36

local _Utils = import '../../utils/helper.libsonnet';

local _Datasets = {
    voc: [
        {},
    ],
};

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
                _Utils.int('data.num_records', 'Records Number', def=0, ddd=true, readonly=true),
                _Utils.int('data.num_classes', 'Classes Number', def=0, ddd=true, readonly=true),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.floatarray('data.normalize.mean', 'Mean', def=[0.5, 0.5, 0.5], ddd=true, readonly=true),
                _Utils.floatarray('data.normalize.std', 'Std', def=[1, 1, 1], ddd=true, readonly=true),
                _Utils.int('data.normalize.div_value', 'Div Value', def=255, ddd=true, readonly=true),
            ],
        },
        // _Utils.stringarray('details.name_seq', 'Name Seq', def=[], ddd=true, width=600, readonly=true),
        // _Utils.intarray('details.color_list', 'Color List', def=[[]], ddd=true, width=600, readonly=true),
    ] + (if std.objectHas(_Datasets, _Utils.dataset_name) then _Datasets[_Utils.dataset_name]
         else []),
}
