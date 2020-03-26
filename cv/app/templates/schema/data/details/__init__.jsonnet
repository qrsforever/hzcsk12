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
                _Utils.image('_k12.data.sample.n0', 'x', def='unkown', ddd=true, width=80, height=80),
                _Utils.image('_k12.data.sample.n1', 'x', def='unkown', ddd=true, width=80, height=80),
                _Utils.image('_k12.data.sample.n2', 'x', def='unkown', ddd=true, width=80, height=80),
                _Utils.image('_k12.data.sample.n3', 'x', def='unkown', ddd=true, width=80, height=80),
                _Utils.image('_k12.data.sample.n4', 'x', def='unkown', ddd=true, width=80, height=80),
                _Utils.image('_k12.data.sample.n5', 'x', def='unkown', ddd=true, width=80, height=80),
                _Utils.image('_k12.data.sample.n6', 'x', def='unkown', ddd=true, width=80, height=80),
                _Utils.image('_k12.data.sample.n7', 'x', def='unkown', ddd=true, width=80, height=80),
                _Utils.image('_k12.data.sample.n8', 'x', def='unkown', ddd=true, width=80, height=80),
                _Utils.image('_k12.data.sample.n9', 'x', def='unkown', ddd=true, width=80, height=80),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.string('_k12.data.dataset_name', 'Dataset Name', def=_Utils.dataset_name, readonly=true),
                _Utils.string('data.data_dir', 'Dataset Path', def='/datasets/' + _Utils.dataset_name, ddd=true, readonly=true),
            ] + (
                if _Utils.task == 'det'
                then
                    if _Utils.dataset_name == 'VOC07+12_DET'
                    then
                        [_Utils.bool('val.use_07_metric', '07 Metric', def=true, readonly=true)]
                    else
                        []
                else []
            ),
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
        _Utils.stringarray('_k12.detail.name_seq', 'Name Seq', def=[], ddd=true, width=600, readonly=true),
        _Utils.image('_k12.detail.data.labelshist', 'Labels Hist', def='labels_hist.png', width=300, height=300),
    ],
}
