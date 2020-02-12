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
            _Utils.string('task', 'Task', def=_Utils.task, readonly=true, tips='task type'),
            _Utils.string('method', 'Method', def=_Utils.method, readonly=true),
        ],
    },
    {
        type: 'H',
        objs: [
            _Utils.float('data.sampling.test_size', 'Test Size', def=0.25, ddd=true),
            _Utils.int('data.sampling.random_state', 'Random State', def=1, ddd=true),
            _Utils.bool('data.sampling.shuffle', 'Shuffle', def=true, ddd=true),
        ],
    },
] + (import 'details/__init__.jsonnet').get()
