// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-22 19:23

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.string('task', 'Task', def=_Utils.task, readonly=true, tips='task type'),
            _Utils.string('data.dataset', 'Dataset Name', def=_Utils.dataset_name, readonly=true),
            _Utils.int('data.jobs', 'Jobs', min=1, max=_Utils.num_cpu, def=4, tips='the numbers of subprocesses for loading dataset'),
        ],
    },
]
