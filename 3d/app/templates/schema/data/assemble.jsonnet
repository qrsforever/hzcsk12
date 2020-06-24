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
            _Utils.string('data.dataset_root', 'Dataset Root', def=_Utils.dataset_root, readonly=true),
            _Utils.string('data.dataset_name', 'Dataset Name', def=_Utils.dataset_name, readonly=true),
        ],
    },
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Loader', cn: self.en },
                type: '_ignore_',
                objs: [(import 'loader/__init__.jsonnet').get('data.dataset_loader')],
            },
        ],
    },
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Transform', cn: self.en },
                type: '_ignore_',
                objs: [(import 'transforms/__init__.jsonnet').get('data.transforms')],
            },
        ],
    },
]
