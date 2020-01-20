// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-17 16:57

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.string('_k12.task', 'Task', def=_Utils.task, readonly=true),
                _Utils.string('_k12.dataset', 'Dataset', def=_Utils.dataset_name, readonly=true),
            ],
        },
        {
            _id_: '_k12.sampler.mode',
            name: { en: 'Mode', cn: self.en },
            type: 'string-enum-trigger',
            objs: [
                {
                    name: { en: 'gpu', cn: self.en },
                    value: 'gpu',
                    trigger: {
                        type: 'H',
                        objs: [
                            _Utils.int('affinity.n_gpu', 'Num GPU', def=1),
                            _Utils.int('affinity.gpu_per_run', 'GPU Per Run', def=1),
                        ],
                    },
                },
                {
                    name: { en: 'cpu', cn: self.en },
                    value: 'cpu',
                    trigger: {},
                },
                {
                    name: { en: 'alternating', cn: self.en },
                    value: 'alternating',
                    trigger: {
                        type: 'H',
                        objs: [
                            _Utils.bool('affinity.alternating', 'Alternating', def=true, readonly=true),
                        ],
                    },
                },
                {
                    name: { en: 'serial', cn: self.en },
                    value: 'serial',
                    trigger: {},
                },
            ],
            default: self.objs[0].value,
        },
        {
            type: 'H',
            objs: [
                _Utils.int('affinity.n_cpu_core', 'Num CPU', def=2),
                _Utils.int('affinity.cpu_per_run', 'CPU Per Run', def=1),
                _Utils.int('affinity.cpu_per_worker', 'Per Worker', def=1),
            ],
        },
        {
            _id_: 'affinity.async_sample',
            name: { en: 'Async', cn: self.en },
            type: 'bool-trigger',
            objs: [
                {
                    value: true,
                    trigger: {
                        type: 'H',
                        objs: [
                            _Utils.int('algo.updates_per_sync', 'Sample Updates', def=1),
                            _Utils.int('affinity.sample_gpu_per_run', 'Sample GPU Per Run', def=1),
                            _Utils.bool('affinity.optim_sample_share_gpu', 'Sample Optim', def=false),
                        ],
                    },
                },
                {
                    value: false,
                    trigger: {},
                },
            ],
        },
    ],
}
