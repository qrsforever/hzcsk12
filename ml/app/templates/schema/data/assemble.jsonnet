// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-11 23:22


local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.string('task', 'Task', def=_Utils.task, ddd=false, readonly=true, tips='task type'),
            _Utils.string('method', 'Method', def=_Utils.method, ddd=false, readonly=true),
            _Utils.bool('data.pca2D', 'PCA 2D', def=false, tips='PCA reduce data to 2 features'),
        ],
    },
    {
        type: 'H',
        objs: [
            _Utils.float('data.sampling.test_size', 'Test Size', def=0.25, max=0.9),
            _Utils.int('data.sampling.random_state', 'Random State', def=42),
            _Utils.bool('data.sampling.shuffle', 'Shuffle', def=true),
        ],
    },
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Details', cn: self.en },
                type: '_ignore_',
                objs: (import 'details/__init__.jsonnet').get(),
            },
        ],
    },
    // {
    //     type: 'H',
    //     objs: [
    //         _Utils.bool('_k12.dev_mode', 'Develop Mode', def=false),
    //     ],
    // },
]
