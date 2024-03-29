// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 15:23

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.string('_k12.data.dataset_name',
                          'Dataset Name',
                          def=_Utils.dataset_name,
                          readonly=true,
                          tips='dataset name'),
        ],
    },
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Reader', cn: self.en },
                type: '_ignore_',
                objs: [
                    (import 'readers/__init__.jsonnet').get(),
                ],
            },
            {
                name: { en: 'Sampler', cn: self.en },
                type: '_ignore_',
                objs: [
                    _Utils.int('data_loader.num_workers',
                               'Num Workers',
                               def=0,
                               min=0,
                               max=_Utils.num_cpu,
                               tips='specify the subprocesses num for data loading'),
                    (import 'samplers/__init__.jsonnet').get('data_loader'),
                ],
            },
        ],
    },
]
