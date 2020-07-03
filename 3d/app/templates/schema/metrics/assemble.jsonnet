// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-22 18:00

local _Utils = import '../utils/helper.libsonnet';

[
    _Utils.int('metrics.log_freq', 'Log Frequency', def=100),
    {
        name: { en: 'Phase', cn: self.en },
        type: 'navigation',
        objs: [
            {
                name: { en: 'Train', cn: self.en },
                type: '_ignore_',
                objs: import 'train.libsonnet',
            },
            {
                name: { en: 'Evaluate', cn: self.en },
                type: '_ignore_',
                objs: import 'evaluate.libsonnet',
            },
            {
                name: { en: 'Pridict', cn: self.en },
                type: '_ignore_',
                objs: import 'predict.libsonnet',
            },
        ],
    },
]
