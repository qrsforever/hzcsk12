// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-27 16:56

local _Utils = import '../utils/helper.libsonnet';

local _phase_train =
    if 'classifier' == _Utils.task
    then
        import 'classifier.libsonnet'
    else if 'regressor' == _Utils.task
    then
        import 'regressor.libsonnet'
    else if 'cluster' == _Utils.task
    then
        import 'cluster.libsonnet'
    else [];

[
    {
        name: { en: 'Phase', cn: self.en },
        type: 'navigation',
        objs: [
            {
                name: { en: 'Train', cn: self.en },
                type: '_ignore_',
                objs: _phase_train,
            },
            {
                name: { en: 'Evaluate', cn: self.en },
                type: '_ignore_',
                objs: [],
            },
            {
                name: { en: 'Pridict', cn: self.en },
                type: '_ignore_',
                objs: [],
            },
        ],
    },
]
