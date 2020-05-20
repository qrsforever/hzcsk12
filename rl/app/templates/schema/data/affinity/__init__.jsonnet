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
                            _Utils.int('affinity.n_gpu', 'Num GPU', def=_Utils.num_gpu, readonly=true),
                            _Utils.int(
                                'affinity.gpu_per_run',
                                'GPU Per Run',
                                def=1,
                                tips='specify gpus to use per task'
                            ),
                            _Utils.bool('affinity.alternating', 'Alternating', def=false, readonly=true),
                        ],
                    },
                },
                {
                    name: { en: 'cpu', cn: self.en },
                    value: 'cpu',
                    trigger: {},
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
                _Utils.int('affinity.n_cpu_core', 'Num CPU', def=_Utils.num_cpu, readonly=true),
                _Utils.int('affinity.cpu_per_run',
                           'CPU Per Run',
                           min=1,
                           max=_Utils.num_cpu,
                           def=1,
                           tips='specify how cpu cores used per task'),
                _Utils.int('affinity.cpu_per_worker',
                           'Per Worker',
                           min=1,
                           max=_Utils.num_cpu,
                           def=1,
                           tips='specify how cpu cores used per sampler worker'),
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
                            _Utils.int('affinity.sample_gpu_per_run',
                                       'SGR',
                                       min=0,
                                       max=_Utils.num_gpu,
                                       def=0,
                                       tips='number of action-server GPUs per task'),
                            _Utils.bool('affinity.optim_sample_share_gpu',
                                        'Sample Optim',
                                        def=false,
                                        tips='whether to use same GPU(s) for both training and sampling'),
                        ],
                    },
                },
                {
                    value: false,
                    trigger: {},
                },
            ],
            default: false,
            // readonly: true,
        },
    ],
}
