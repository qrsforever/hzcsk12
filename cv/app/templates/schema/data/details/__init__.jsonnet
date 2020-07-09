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
                _Utils.string('data.data_dir', 'Dataset Path', def='/datasets/' + _Utils.dataset_name, ddd=true, readonly=true),
            ] + (
                if _Utils.task == 'det'
                then
                    [_Utils.bool('val.use_07_metric', '07 Metric', def=true, readonly=true)]
                else []
            ),
        },
    ] + (if _Utils.task == 'gan'
         then []
         else [
             {
                 type: 'H',
                 objs: [
                     _Utils.int('data.num_records', 'Records Number', def=0, ddd=true, readonly=true),
                     _Utils.int('data.num_classes', 'Classes Number', def=0, ddd=true, readonly=true),
                 ],
             },
         ]) + [
        {
            type: 'H',
            objs: [
                _Utils.floatarray('data.normalize.mean', 'Mean', def=[0.5, 0.5, 0.5], ddd=true, readonly=true),
                _Utils.floatarray('data.normalize.std', 'Std', def=[0.5, 0.5, 0.5], ddd=true, readonly=true),
            ],
        },
        _Utils.stringarray('_k12.detail.name_seq', 'Name Seq', def=[], ddd=true, width=600, readonly=true),
        {
            name: { en: 'Data Samples', cn: self.en },
            type: 'H',
            objs: [
                _Utils.sampleimage('_k12.data.sample.n0', 'label 0', def='unkown', ddd=true, width=80, height=80),
                _Utils.sampleimage('_k12.data.sample.n1', 'label 1', def='unkown', ddd=true, width=80, height=80),
                _Utils.sampleimage('_k12.data.sample.n2', 'label 2', def='unkown', ddd=true, width=80, height=80),
                _Utils.sampleimage('_k12.data.sample.n3', 'label 3', def='unkown', ddd=true, width=80, height=80),
                _Utils.sampleimage('_k12.data.sample.n4', 'label 4', def='unkown', ddd=true, width=80, height=80),
                _Utils.sampleimage('_k12.data.sample.n5', 'label 5', def='unkown', ddd=true, width=80, height=80),
                _Utils.sampleimage('_k12.data.sample.n6', 'label 6', def='unkown', ddd=true, width=80, height=80),
                _Utils.sampleimage('_k12.data.sample.n7', 'label 7', def='unkown', ddd=true, width=80, height=80),
                _Utils.sampleimage('_k12.data.sample.n8', 'label 8', def='unkown', ddd=true, width=80, height=80),
                _Utils.sampleimage('_k12.data.sample.n9', 'label 9', def='unkown', ddd=true, width=80, height=80),
            ],
        },
        {
            name: { en: 'Data Analysis', cn: self.en },
            type: 'H',
            objs: [
                _Utils.sampleimage('_k12.detail.data.analysis.labelshist', 'Labels Hist', def='labels_hist.png', width=500, height=500),
            ],
        },
    ],
}
